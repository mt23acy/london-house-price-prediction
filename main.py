import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import gradio as gr
import traceback
import re

# --- Constants ---

TARGET_YEAR_FOR_SEPARATION = 2024 # Year excluded from training
MODEL_NAME = "XGBoost Regressor (Address Lookup)"
PREDICTION_HORIZON_YEARS = 3 # How many years into the future to predict for the trend plot

# --- Filter Warnings ---

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Variables (for Gradio state) ---

xgb_model_viz = None
x_scaler_viz = None
category_mappings_viz = None # To store label encoding mappings
training_columns_viz = None # To store feature order/names after encoding
categorical_cols_viz = None # List of original categorical column names
default_categorical_values_viz = None # Modes for non-address categoricals (encoded)
df_cleaned_viz = None # To store cleaned data for historical lookups & dropdowns (original data)
train_df_encoded_viz = None # To store the encoded training data (used for finding feature modes)
price_min_filter_viz = 10_000
price_max_filter_viz = 20_000_000
unique_streets_viz = [] # List of unique street names for dropdown

# --- Address Cleaning Function ---

def clean_address_string(addr_str):
    if pd.isna(addr_str): return ''
    addr_str = str(addr_str).upper()
    addr_str = re.sub(r'[^\w\s]', '', addr_str); addr_str = re.sub(r'\s+', ' ', addr_str).strip()
    addr_str = addr_str.replace(' ROAD', ' RD').replace(' STREET', ' ST').replace(' AVENUE', ' AVE')
    addr_str = addr_str.replace(' COURT', ' CT').replace(' LANE', ' LN').replace(' SQUARE', ' SQ')
    addr_str = addr_str.replace(' CLOSE', ' CL').replace(' DRIVE', ' DR')
    addr_str = addr_str.replace(' FLAT ', 'FLT ').replace(' APARTMENT ', 'APT ')
    return addr_str

# --- Setup Function (Data Loading, Preprocessing, Training) ---

def setup_data_and_model_for_viz():
    global xgb_model_viz, x_scaler_viz, category_mappings_viz, training_columns_viz, categorical_cols_viz
    global default_categorical_values_viz, df_cleaned_viz, price_min_filter_viz, price_max_filter_viz, unique_streets_viz
    global train_df_encoded_viz # Added global declaration

    print("--- Running Setup: Loading Data ---")
    try:
        df = pd.read_csv("london_merged_df_filtered.csv")
        print(f"Successfully loaded merged data. Initial shape: {df.shape}")
    except FileNotFoundError:
        print("Error: london_merged_df_filtered.csv not found.")
        return False

    cols_to_drop_initial = ['postcode_cleaned', 'ppd_address_key', 'epc_address_key',
                           'lodgement_date','lmk_key','epc_built_form','epc_age_band',
                           'record status - monthly file only', "Transaction unique identifier"]
    df = df.drop(columns=[col for col in cols_to_drop_initial if col in df.columns], errors='ignore')

    print("\n--- Initial Cleaning & Filtering ---")
    required_cols = ['postcode', 'PAON', 'street', 'epc_floor_area', 'epc_habitable_rooms']
    if not all(col in df.columns for col in required_cols):
         print(f"Error: Missing one or more required columns: {required_cols}")
         return False
    initial_rows = len(df)
    df = df.dropna(subset=required_cols)
    print(f"Dropped {initial_rows - len(df)} rows missing critical features.")

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df['price'] = df['price'].astype(float)
    df = df[(df["price"] >= price_min_filter_viz) & (df["price"] <= price_max_filter_viz)]
    print(f"Filtered prices. Shape after filtering: {df.shape}")
    if df.empty: print("Error: DataFrame empty after price filtering."); return False

    print("\n--- Feature Engineering (Date) ---")
    df["date of transfer"] = pd.to_datetime(df["date of transfer"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["date of transfer"])
    if df.empty: print("Error: DataFrame empty after date filtering."); return False
    df['year']    = df['date of transfer'].dt.year
    df["month"]   = df['date of transfer'].dt.month
    df["day"]     = df['date of transfer'].dt.day
    df["weekday"] = df['date of transfer'].dt.weekday
    df["quarter"] = df['date of transfer'].dt.quarter
    print("Extracted date features.")

    # Store cleaned data BEFORE encoding
    df_cleaned_viz = df.copy()
    df_cleaned_viz['street_lower'] = df_cleaned_viz['street'].str.lower().str.strip()
    # Store unique streets (original case) for display/lookup consistency
    unique_streets_viz = sorted(df_cleaned_viz['street'].dropna().unique().tolist())
    print(f"Found {len(unique_streets_viz)} unique street names.")

    # --- Apply Label Encoding (on a copy for training prep) ---
    df_encoded = df.copy()
    # Identify all potential categorical columns including PAON/SAON/Postcode
    categorical_cols_viz = df_encoded.select_dtypes(include=['object']).columns.tolist()
    for col in ['postcode', 'PAON', 'SAON']: # Ensure these are treated as categorical
         if col not in categorical_cols_viz and col in df_encoded.columns:
             categorical_cols_viz.append(col)
             df_encoded[col] = df_encoded[col].astype(str) # Convert to string before category
    categorical_cols_viz = [col for col in categorical_cols_viz if col in df_encoded.columns and col != 'date of transfer']

    print(f"\n--- Applying Label Encoding for Training ---")
    print(f"Columns to Label Encode: {categorical_cols_viz}")
    category_mappings_viz = {}
    for col in categorical_cols_viz:
        # Ensure category exists before encoding
        # Use .astype('category').cat.set_categories to handle potential missing categories more robustly
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].astype(str) # Ensure string type
            try:
                df_encoded[col] = df_encoded[col].astype('category')
                category_mappings_viz[col] = df_encoded[col].cat.categories
                df_encoded[col] = df_encoded[col].cat.codes
            except Exception as e:
                print(f"Error encoding column {col}: {e}")
                df_encoded[col] = -999 # Assign a default code if encoding fails


    # --- Separate Target Year ---
    print(f"\n--- Separating Data for Training (Excluding {TARGET_YEAR_FOR_SEPARATION}) ---")
    if TARGET_YEAR_FOR_SEPARATION in df_encoded['year'].unique():
        df_for_training_encoded = df_encoded[df_encoded['year'] != TARGET_YEAR_FOR_SEPARATION].copy()
    else:
        df_for_training_encoded = df_encoded.copy()

    # Store the encoded training data for later use (e.g., finding modes for prediction)
    train_df_encoded_viz = df_for_training_encoded.copy()

    # --- Prepare Training Data ---
    train_df = df_for_training_encoded.drop(columns=['date of transfer'])
    # train_df = train_df.replace(-1, -999) # Handled during encoding now if category not found
    train_df.fillna(0, inplace=True) # Fill any remaining NaNs

    # Calculate Defaults (Modes from encoded training data)
    print("\n--- Calculating Default Values for Prediction ---")
    default_categorical_values_viz = {}
    temp_train_df_for_defaults = train_df.drop(columns=['price'], errors='ignore')

    # Ensure all columns that *were* encoded are considered for defaults, even if not explicitly 'categorical_cols_viz'
    # (e.g. numeric cols like floor_area might be in train_df, but not in categorical_cols_viz)
    for col in temp_train_df_for_defaults.columns:
         mode_val = temp_train_df_for_defaults[col].mode()
         if not mode_val.empty:
             default_categorical_values_viz[col] = mode_val[0]
         else:
             default_categorical_values_viz[col] = 0 # Default to 0 if no mode (e.g., all NaNs in original, or empty category)


    # --- Data Splitting & Scaling ---
    X = train_df.drop(['price'], axis=1)
    y = train_df['price']
    training_columns_viz = X.columns.tolist() # Store the exact column order for prediction
    n_features = len(training_columns_viz)
    print(f"Number of features after label encoding: {n_features}")

    if X.empty or y.empty: print("Error: Training features/target empty."); return False
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape (X_train): {X_train.shape}")

    print("\n--- Scaling Features ---")
    x_scaler_viz = StandardScaler()
    X_train_scaled = x_scaler_viz.fit_transform(X_train)
    X_test_scaled = x_scaler_viz.transform(X_test)
    print("Features scaled using StandardScaler.")

    # --- Train XGBoost Model ---
    print(f"\n--- Defining and Training {MODEL_NAME} Model ---")
    xgb_model_viz = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.05,
        max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse'
    )
    print("\nStarting model training...")
    eval_set = [(X_test_scaled, y_test)]
    # Removed verbose=False to potentially see training progress
    xgb_model_viz.fit(X_train_scaled, y_train, eval_set=eval_set) # verbose is True by default

    print("\nModel training complete.")

    print("\n--- Setup Complete ---")
    return True

# --- Gradio Helper Functions ---

def update_paon_dropdown(street_input):
    """Dynamically update PAON dropdown choices based on street input."""
    global df_cleaned_viz
    postcode_reset = gr.update(choices=[], value=None, interactive=False) # Reset postcode dropdown
    if df_cleaned_viz is None or street_input is None or street_input.strip() == '':
        print("Clearing PAON and Postcode dropdowns.")
        # Return updates for BOTH dropdowns
        return gr.update(choices=[], value=None), postcode_reset

    street_input_clean = street_input.strip().lower()
    matching_paons = df_cleaned_viz[df_cleaned_viz['street_lower'] == street_input_clean]['PAON'].dropna().unique()
    sorted_paons_str = []
    if len(matching_paons) > 0:
        try:
            numeric_paons, non_numeric_paons = [], []
            for p in matching_paons:
                p_str = str(p).strip()
                num_part = re.search(r'^(\d+)', p_str)
                if num_part: numeric_paons.append((int(num_part.group(1)), p_str))
                else: non_numeric_paons.append(p_str)
            numeric_paons.sort(key=lambda item: item[0])
            sorted_paons_str = [item[1] for item in numeric_paons] + sorted(non_numeric_paons)
        except Exception as e:
            print(f"PAON sorting error: {e}, falling back to string sort.")
            sorted_paons_str = sorted([str(p) for p in matching_paons])
    # print(f"Found {len(sorted_paons_str)} PAONs for street '{street_input}'.") # Too verbose
    # Return updates for BOTH dropdowns
    return gr.update(choices=sorted_paons_str, value=None, interactive=True), postcode_reset


def update_postcode_dropdown(street_input, paon_input):
    """Dynamically update Postcode dropdown based on street and PAON."""
    global df_cleaned_viz
    if df_cleaned_viz is None or not street_input or not paon_input:
        print("Clearing Postcode dropdown (missing street or PAON).")
        return gr.update(choices=[], value=None, interactive=False)

    street_input_clean = street_input.strip().lower()
    paon_input_clean = str(paon_input).strip()

    # Filter based on both street and PAON
    matching_postcodes = df_cleaned_viz[
        (df_cleaned_viz['street_lower'] == street_input_clean) &
        (df_cleaned_viz['PAON'].astype(str).str.strip() == paon_input_clean)
    ]['postcode'].dropna().unique()

    sorted_postcodes = sorted([str(pc) for pc in matching_postcodes])
    # print(f"Found {len(sorted_postcodes)} Postcodes for '{paon_input_clean} {street_input}'.") # Too verbose

    # Enable postcode dropdown only if postcodes are found
    interactive_state = len(sorted_postcodes) > 0
    # Select the first postcode automatically if only one exists
    default_value = sorted_postcodes[0] if len(sorted_postcodes) == 1 else None

    return gr.update(choices=sorted_postcodes, value=default_value, interactive=interactive_state)


def get_prices_and_predict(street_input, paon_input, postcode_input):
    """Fetch historical data, plot it, show table, and predict current price using all three inputs."""
    global df_cleaned_viz, xgb_model_viz, x_scaler_viz, category_mappings_viz, training_columns_viz
    global categorical_cols_viz, default_categorical_values_viz, price_min_filter_viz, price_max_filter_viz

    # --- Initialize Outputs ---
    historical_plot = None
    predicted_price_str = "Prediction could not be made."
    historical_table_data = None # Initialize table data to None (will be pandas DataFrame or None)

    # --- Input Validation ---
    if not all([street_input, paon_input, postcode_input]):
        return historical_plot, "Please select Street, House Number/Name, AND Postcode.", historical_table_data # Return all outputs
    if df_cleaned_viz is None or xgb_model_viz is None or x_scaler_viz is None:
        return historical_plot, "System not ready. Please wait for setup.", historical_table_data # Return all outputs

    street_input_clean = street_input.strip().lower()
    paon_input_clean = str(paon_input).strip()
    postcode_input_clean = str(postcode_input).strip().upper() # Clean postcode input
    print(f"\nProcessing Request - Street: '{street_input_clean}', PAON: '{paon_input_clean}', Postcode: '{postcode_input_clean}'")

    # --- Find Historical Data ---
    historical_data = df_cleaned_viz[
        (df_cleaned_viz['street_lower'] == street_input_clean) &
        (df_cleaned_viz['PAON'].astype(str).str.strip() == paon_input_clean) &
        (df_cleaned_viz['postcode'].str.upper() == postcode_input_clean) # Match postcode too
    ].copy()

    latest_property_data = None

    if not historical_data.empty:
        print(f"Found {len(historical_data)} historical transaction(s).")
        historical_data = historical_data.sort_values('date of transfer')
        latest_property_data = historical_data.iloc[-1:].copy()

        # --- Create Historical Plot ---
        try:
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.plot(historical_data['date of transfer'], historical_data['price'], marker='o', linestyle='-')
            ax_hist.set_xlabel("Date of Transfer")
            ax_hist.set_ylabel("Price (£)")
            title = f"Historical Sales: {paon_input_clean} {street_input.strip().title()}, {postcode_input_clean}"
            ax_hist.set_title(title)
            ax_hist.grid(True)
            formatter = FuncFormatter(lambda x, p: f'£{x:,.0f}')
            ax_hist.yaxis.set_major_formatter(formatter)
            plt.setp(ax_hist.get_xticklabels(), rotation=30, ha='right')
            fig_hist.tight_layout()
            historical_plot = fig_hist
            print("Generated historical plot.")
        except Exception as e:
            print(f"Error generating historical plot: {e}")
            traceback.print_exc()
            historical_plot = None # Ensure it's None if plotting fails

        # --- Prepare Historical Data Table ---
        try:
            historical_table_data = historical_data[['date of transfer', 'price']].copy()
            historical_table_data['date of transfer'] = historical_table_data['date of transfer'].dt.strftime('%Y-%m-%d')
            historical_table_data['price'] = historical_table_data['price'].apply(lambda x: f'£{x:,.2f}')
            historical_table_data.rename(columns={'date of transfer': 'Date of Transfer', 'price': 'Price Paid (£)'}, inplace=True)
            print("Prepared historical table data.")
        except Exception as e:
            print(f"Error preparing historical table data: {e}")
            traceback.print_exc()
            historical_table_data = None # Ensure it's None if table prep fails

    else:
        print("No historical data found for this exact address/postcode.")
        predicted_price_str = "No historical data found for this exact address."
        # historical_plot and historical_table_data remain None

    # --- Predict Current Price ---
    if latest_property_data is not None:
        try:
            # 1. Prepare Input Row
            input_df = latest_property_data.reset_index(drop=True)
            now = datetime.now()
            input_df.loc[0, 'year'] = now.year; input_df.loc[0, 'month'] = now.month
            input_df.loc[0, 'day'] = now.day; input_df.loc[0, 'weekday'] = now.weekday()
            input_df.loc[0, 'quarter'] = (now.month - 1) // 3 + 1

            # 2. Apply Consistent Label Encoding and Handle Unseen Categories/Missing Columns
            processed_input_row = {}
            for col in training_columns_viz: # Iterate through columns expected by the model
                 if col in input_df.columns:
                     if col in categorical_cols_viz and col in category_mappings_viz:
                         input_val_str = str(input_df.loc[0, col])
                         if input_val_str in category_mappings_viz[col]:
                              processed_input_row[col] = category_mappings_viz[col].get_loc(input_val_str)
                         else:
                              # print(f"  Warning: Category '{input_val_str}' in '{col}' unseen during prediction. Using default.") # Too verbose
                              processed_input_row[col] = default_categorical_values_viz.get(col, -999) # Use default from training or -999
                     else: # Not a known categorical column or no mapping, use value directly
                         processed_input_row[col] = input_df.loc[0, col]
                 elif col in default_categorical_values_viz:
                      # print(f"  Info: Column '{col}' missing in input, using default {default_categorical_values_viz[col]}.") # Too verbose
                      processed_input_row[col] = default_categorical_values_viz[col]
                 else:
                     # print(f"  Warning: Column '{col}' missing in input and no default. Using 0.") # Too verbose
                     processed_input_row[col] = 0 # Fallback if column is entirely new and no default was calculated

            input_aligned = pd.DataFrame([processed_input_row], columns=training_columns_viz)


            # 3. Handle -1/-999 codes / Fill NaNs (although fillna(0) was used earlier)
            # Let's explicitly handle -1 codes which label encoding might produce for NaNs if not careful
            # Replace any remaining -1s (from label encoding NaNs) with -999 for XGBoost
            input_aligned = input_aligned.replace(-1, -999)
            input_aligned.fillna(0, inplace=True) # Just in case


            # 4. Scale Features
            input_scaled = x_scaler_viz.transform(input_aligned)

            # 5. Predict
            predicted_price = xgb_model_viz.predict(input_scaled)[0]

            # 6. Format Output
            print(f"Raw Predicted Current Price: {predicted_price}")
            if np.isfinite(predicted_price) and predicted_price > 0:
                # Apply reasonable bounds based on the filtered price range
                bounded_price = max(price_min_filter_viz * 0.5, min(predicted_price, price_max_filter_viz * 2.0)) # Adjusted bounds slightly
                # if abs(bounded_price - predicted_price) > 1000: # Only print if significant change
                #     print(f"Bounded prediction from {predicted_price:,.0f} to {bounded_price:,.0f}")
                predicted_price_str = f"£{bounded_price:,.2f}"
            elif predicted_price <= 0: predicted_price_str = "Prediction non-positive."
            else: predicted_price_str = "Prediction failed (NaN/Inf)."

        except Exception as e:
            print(f"Error during prediction step: {e}")
            traceback.print_exc()
            predicted_price_str = f"Error predicting current price."

    # Return the plot, the prediction string, AND the historical table data
    return historical_plot, predicted_price_str, historical_table_data


def generate_london_trend_plot():
    """Generates a plot showing historical London median prices and future predictions."""
    global df_cleaned_viz, xgb_model_viz, x_scaler_viz, category_mappings_viz, training_columns_viz
    global default_categorical_values_viz, train_df_encoded_viz, PREDICTION_HORIZON_YEARS

    if df_cleaned_viz is None or xgb_model_viz is None or x_scaler_viz is None or train_df_encoded_viz is None:
        print("System not ready for trend plot. Setup incomplete.")
        # Return an empty plot placeholder
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Setup incomplete. Cannot generate plot.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title("London Property Price Trend (Setup Pending)")
        ax.axis('off')
        return fig

    print("\n--- Generating London Trend Plot ---")
    try:
        # --- 1. Calculate Historical Median Prices ---
        historical_monthly_median = df_cleaned_viz.groupby(['year', 'month'])['price'].median().reset_index()
        historical_monthly_median['date'] = pd.to_datetime(historical_monthly_median[['year', 'month']].assign(day=1))
        historical_monthly_median = historical_monthly_median.sort_values('date')
        print(f"Calculated historical median prices for {len(historical_monthly_median)} months.")

        if historical_monthly_median.empty:
             print("No historical data to plot trend.")
             fig, ax = plt.subplots()
             ax.text(0.5, 0.5, "No historical data to plot trend.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             ax.set_title("London Property Price Trend (No Data)")
             ax.axis('off')
             return fig


        # --- 2. Prepare Base Features for Future Prediction ---
        # Use the mode of features from the *encoded training data* as a base
        # Exclude date columns and price, as these will be generated for future dates
        features_for_mode = train_df_encoded_viz.drop(columns=['price', 'year', 'month', 'day', 'weekday', 'quarter', 'date of transfer'], errors='ignore')
        # Calculate mode for all remaining columns
        base_features = features_for_mode.mode().iloc[0].to_dict() # Get the first row of modes as a dictionary
        print("Calculated mode features from training data for prediction base.")
        # print("Base Features:", base_features) # Debug print


        # --- 3. Generate Future Dates ---
        last_historical_date = historical_monthly_median['date'].max()
        start_future_date = last_historical_date + pd.offsets.MonthBegin(1)
        future_dates = [start_future_date + pd.offsets.MonthBegin(i) for i in range(PREDICTION_HORIZON_YEARS * 12)]
        print(f"Generated {len(future_dates)} future dates starting from {start_future_date.strftime('%Y-%m-%d')}.")

        # --- 4. Create Future Dataframe for Prediction ---
        future_data_list = []
        for date in future_dates:
            future_row = base_features.copy() # Start with mode features
            # Update date-related features
            future_row['year'] = date.year
            future_row['month'] = date.month
            future_row['day'] = date.day # Using the 1st of the month
            future_row['weekday'] = date.weekday()
            future_row['quarter'] = (date.month - 1) // 3 + 1
            # Add any other calculated/derived date features if used in training
            # e.g., days_since_start = (date - training_start_date).days # if used

            # Ensure all columns in training_columns_viz are present
            full_future_row = {}
            for col in training_columns_viz:
                 full_future_row[col] = future_row.get(col, default_categorical_values_viz.get(col, 0)) # Use future_row val, then default, then 0

            future_data_list.append(full_future_row)

        future_df_raw = pd.DataFrame(future_data_list, columns=training_columns_viz) # Ensure correct column order
        # future_df_raw = future_df_raw.replace(-1, -999) # Handled by default values if needed
        future_df_raw.fillna(0, inplace=True) # Ensure no NaNs remain

        # print("Raw future data for prediction (first 5 rows):") # Debug print
        # print(future_df_raw.head())


        # --- 5. Scale Future Data ---
        future_scaled = x_scaler_viz.transform(future_df_raw)
        print(f"Scaled future data ({future_scaled.shape}).")

        # --- 6. Predict Future Prices ---
        predicted_future_prices = xgb_model_viz.predict(future_scaled)
        future_df_raw['predicted_price'] = predicted_future_prices
        future_df_raw['date'] = future_dates # Add the actual date objects

        # --- 7. Combine and Plot ---
        fig_trend, ax_trend = plt.subplots(figsize=(12, 6))

        # Plot historical data
        ax_trend.plot(historical_monthly_median['date'], historical_monthly_median['price'], marker='o', linestyle='-', label='Historical Median Price')

        # Plot future predictions (dotted line)
        ax_trend.plot(future_df_raw['date'], future_df_raw['predicted_price'], linestyle=':', color='red', label='Predicted Price Trend')

        # Add a vertical line at the separation point
        if last_historical_date:
             ax_trend.axvline(last_historical_date, color='gray', linestyle='--', label=f'End of Historical Data ({last_historical_date.year})')

        ax_trend.set_xlabel("Date")
        ax_trend.set_ylabel("Price (£)")
        ax_trend.set_title(f"London Property Price Trend: Historical Median and {PREDICTION_HORIZON_YEARS}-Year Prediction")
        ax_trend.grid(True)
        formatter = FuncFormatter(lambda x, p: f'£{x:,.0f}')
        ax_trend.yaxis.set_major_formatter(formatter)
        plt.setp(ax_trend.get_xticklabels(), rotation=30, ha='right')
        ax_trend.legend()
        fig_trend.tight_layout()

        print("Generated London trend plot.")
        return fig_trend

    except Exception as e:
        print(f"Error generating London trend plot: {e}")
        traceback.print_exc()
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Error generating plot: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')
        ax.set_title("London Property Price Trend (Error)")
        ax.axis('off')
        return fig


# --- Run Setup ---

setup_successful = setup_data_and_model_for_viz()

# --- Create and Launch Gradio Interface ---

if setup_successful:
    print("\n--- Launching Gradio Address Lookup & Trend Interface ---")

    # Define the Address Lookup tab content
    address_lookup_block = gr.Blocks()
    with address_lookup_block:
        gr.Markdown("# UK House Price Lookup & Prediction (XGBoost)")
        gr.Markdown("Enter a Street Name, select a House Number/Name, then select the Postcode to see historical prices and a current prediction for a specific property.")

        with gr.Row():
            # Changed to Textbox for better filtering before dropdown
            street_txt = gr.Textbox(label="Street Name", placeholder="Start typing street name...", interactive=True)
            paon_dd = gr.Dropdown(label="House Number/Name (PAON)", choices=[], interactive=False) # Start disabled
            postcode_dd = gr.Dropdown(label="Postcode", choices=[], interactive=False) # Start disabled

        predict_btn = gr.Button("Get Prices & Predict Current")

        with gr.Row():
             predicted_price_out = gr.Textbox(label="Predicted Current Price (£)", scale=1)
             historical_table_out = gr.DataFrame(label="Historical Transactions", wrap=True, scale=2)

        historical_plot_out = gr.Plot(label="Historical Prices for this Address")


        # --- Address Lookup Interactions ---
        street_txt.change(
            update_paon_dropdown,
            inputs=street_txt,
            outputs=[paon_dd, postcode_dd]
            )

        paon_dd.change(
            update_postcode_dropdown,
            inputs=[street_txt, paon_dd],
            outputs=postcode_dd
            )

        predict_btn.click(
            get_prices_and_predict,
            inputs=[street_txt, paon_dd, postcode_dd],
            outputs=[historical_plot_out, predicted_price_out, historical_table_out]
        )

    # Define the London Trend tab content
    london_trend_block = gr.Blocks()
    with london_trend_block:
        gr.Markdown("# London Property Price Trend")
        gr.Markdown("Displays the historical median property price trend across London and a future prediction using the trained model.")
        # The plot is generated by calling the function directly when the tab is loaded/defined
        london_trend_plot_out = gr.Plot(
            label="London Property Price Trend",
            value=generate_london_trend_plot # Call the function to generate the initial plot
            )


    # Create the tabbed interface
    demo = gr.TabbedInterface(
        [address_lookup_block, london_trend_block],
        ["Address Lookup", "London Trend"]
    )


    print("Interface setup complete. Launching...")
    demo.launch(show_api=False)
    print("\n--- Gradio App Running ---")

else:
    print("\n--- Setup Failed. Gradio interface will not launch. ---")

print("\n--- Full Script Finished ---")
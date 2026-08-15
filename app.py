import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Porter Delivery Time Prediction",
    page_icon="🚚",
    layout="centered"
)


# =========================================================
# TENSORFLOW CONFIGURATION
# =========================================================

# Limit TensorFlow CPU threads.
# This can help prevent TensorFlow from hanging on some Macs.

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


# =========================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "porter_delivery_model.keras"
    )


@st.cache_resource
def load_scaler():

    return joblib.load(
        "scaler.pkl"
    )


@st.cache_resource
def load_encoder():

    return joblib.load(
        "onehot_encoder.pkl"
    )


@st.cache_resource
def load_features():

    return joblib.load(
        "feature_columns.pkl"
    )


# Load objects
model = load_model()
scaler = load_scaler()
ohe = load_encoder()
feature_columns = load_features()


# =========================================================
# TITLE
# =========================================================

st.title(
    "🚚 Porter Delivery Time Prediction"
)

st.markdown(
    """
    ### AI-Powered Delivery Time Estimator

    Enter the order, restaurant and delivery-partner
    information below to estimate the delivery time using
    an Artificial Neural Network (ANN) regression model.
    """
)

st.divider()


# =========================================================
# ORDER INFORMATION
# =========================================================

st.subheader(
    "📦 Order Information"
)

col1, col2 = st.columns(2)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    market_id = st.number_input(
        "Market ID",
        min_value=0,
        step=1,
        value=1
    )

    order_protocol = st.number_input(
        "Order Protocol",
        min_value=0,
        max_value=7,
        step=1,
        value=1
    )

    total_items = st.number_input(
        "Total Items",
        min_value=1,
        step=1,
        value=2
    )

    num_distinct_items = st.number_input(
        "Number of Distinct Items",
        min_value=1,
        step=1,
        value=2
    )

    min_item_price = st.number_input(
        "Minimum Item Price",
        min_value=0,
        step=10,
        value=50
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    subtotal = st.number_input(
        "Subtotal",
        min_value=0,
        step=50,
        value=300
    )

    max_item_price = st.number_input(
        "Maximum Item Price",
        min_value=0,
        step=10,
        value=150
    )


# =========================================================
# RESTAURANT INFORMATION
# =========================================================

st.subheader(
    "🏪 Restaurant Information"
)


# Get categories from the trained encoder
categories = list(
    ohe.categories_[0]
)


store_primary_category = st.selectbox(
    "Store Primary Category",
    categories
)


# =========================================================
# DELIVERY PARTNER INFORMATION
# =========================================================

st.subheader(
    "🛵 Delivery Partner Information"
)

col1, col2, col3 = st.columns(3)


with col1:

    total_onshift_partners = st.number_input(
        "On-shift Partners",
        min_value=0,
        step=1,
        value=10
    )


with col2:

    total_busy_partners = st.number_input(
        "Busy Partners",
        min_value=0,
        step=1,
        value=5
    )


with col3:

    total_outstanding_orders = st.number_input(
        "Outstanding Orders",
        min_value=0,
        step=1,
        value=5
    )


st.divider()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def make_prediction():

    # =====================================================
    # STEP 1: CREATE NUMERICAL DATAFRAME
    # =====================================================

    numerical_data = pd.DataFrame({

        "market_id": [
            market_id
        ],

        "order_protocol": [
            order_protocol
        ],

        "total_items": [
            total_items
        ],

        "subtotal": [
            subtotal
        ],

        "num_distinct_items": [
            num_distinct_items
        ],

        "min_item_price": [
            min_item_price
        ],

        "max_item_price": [
            max_item_price
        ],

        "total_onshift_partners": [
            total_onshift_partners
        ],

        "total_busy_partners": [
            total_busy_partners
        ],

        "total_outstanding_orders": [
            total_outstanding_orders
        ]

    })


    # =====================================================
    # STEP 2: CREATE CATEGORY DATAFRAME
    # =====================================================

    category_input = pd.DataFrame({

        "store_primary_category": [
            store_primary_category
        ]

    })


    # =====================================================
    # STEP 3: ONE-HOT ENCODE CATEGORY
    # =====================================================

    category_encoded = ohe.transform(
        category_input
    )


    # If encoder returns sparse matrix,
    # convert it to a normal NumPy array.

    if hasattr(
        category_encoded,
        "toarray"
    ):

        category_encoded = (
            category_encoded.toarray()
        )


    # Create DataFrame for encoded features

    category_df = pd.DataFrame(

        category_encoded,

        columns=ohe.get_feature_names_out(
            ["store_primary_category"]
        )

    )


    # =====================================================
    # STEP 4: COMBINE NUMERICAL + CATEGORICAL FEATURES
    # =====================================================

    final_input = pd.concat(

        [
            numerical_data,
            category_df
        ],

        axis=1

    )


    # =====================================================
    # STEP 5: ENSURE EXACT FEATURE ORDER
    # =====================================================

    final_input = final_input.reindex(

        columns=feature_columns,

        fill_value=0

    )


    # =====================================================
    # STEP 6: CHECK FEATURE COUNT
    # =====================================================

    expected_features = scaler.n_features_in_

    actual_features = final_input.shape[1]


    if actual_features != expected_features:

        raise ValueError(
            f"Feature mismatch: "
            f"Expected {expected_features} features "
            f"but received {actual_features}."
        )


    # =====================================================
    # STEP 7: SCALE FEATURES
    # =====================================================

    input_scaled = scaler.transform(
        final_input
    )


    # Convert to float32
    input_scaled = np.asarray(
        input_scaled,
        dtype=np.float32
    )


    # =====================================================
    # STEP 8: CHECK INPUT SHAPE
    # =====================================================

    if input_scaled.shape != (1, 85):

        raise ValueError(
            f"Incorrect input shape: "
            f"{input_scaled.shape}. "
            f"Expected (1, 85)."
        )


    # =====================================================
    # STEP 9: CONVERT TO TENSOR
    # =====================================================

    input_tensor = tf.convert_to_tensor(

        input_scaled,

        dtype=tf.float32

    )


    # =====================================================
    # STEP 10: MODEL PREDICTION
    # =====================================================

    prediction_tensor = model(

        input_tensor,

        training=False

    )


    # Convert TensorFlow tensor to NumPy

    prediction = (
        prediction_tensor.numpy()
    )


    # =====================================================
    # STEP 11: EXTRACT PREDICTED DELIVERY TIME
    # =====================================================

    delivery_time = float(
        prediction[0][0]
    )


    return delivery_time


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(

    "🚚 Predict Delivery Time",

    use_container_width=True

):

    try:

        # -------------------------------------------------
        # SHOW PROGRESS
        # -------------------------------------------------

        with st.spinner(
            "🤖 Calculating delivery time..."
        ):

            delivery_time = (
                make_prediction()
            )


        # -------------------------------------------------
        # SUCCESS MESSAGE
        # -------------------------------------------------

        st.success(
            "✅ Prediction completed successfully!"
        )


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.metric(

            label="Estimated Delivery Time",

            value=f"{delivery_time:.2f} minutes"

        )


        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        if delivery_time < 30:

            st.info(
                "⚡ The estimated delivery time "
                "is relatively short."
            )

        elif delivery_time < 60:

            st.info(
                "🛵 The estimated delivery time "
                "is moderate."
            )

        else:

            st.warning(
                "⏳ The estimated delivery time "
                "is relatively high."
            )


    except Exception as e:

        # -------------------------------------------------
        # ERROR MESSAGE
        # -------------------------------------------------

        st.error(
            "❌ Prediction failed"
        )

        st.exception(e)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader(
    "🤖 Model Information"
)


col1, col2 = st.columns(2)


# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with col1:

    st.write(
        "**Model:** Artificial Neural Network"
    )

    st.write(
        "**Task:** Regression"
    )

    st.write(
        "**Target:** Delivery Time (minutes)"
    )


# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with col2:

    st.write(
        "**Hidden Activation:** ReLU"
    )

    st.write(
        "**Optimizer:** Adam"
    )

    st.write(
        "**Loss:** Mean Squared Error"
    )


# =========================================================
# ANN ARCHITECTURE
# =========================================================

with st.expander(
    "🔍 View ANN Architecture"
):

    st.code(
        """
Input Layer: 85 features
        ↓
Dense Layer: 128 neurons - ReLU
        ↓
Dense Layer: 64 neurons - ReLU
        ↓
Dense Layer: 32 neurons - ReLU
        ↓
Dense Layer: 16 neurons - ReLU
        ↓
Output Layer: 1 neuron
        """,
        language="text"
    )


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Porter Delivery Time Prediction | ANN Regression"
)
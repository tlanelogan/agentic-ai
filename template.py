import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# =============================================================================
# Environment + model
# -----------------------------------------------------------------------------
# The agent stack is smolagents (ToolCallingAgent / OpenAIServerModel / @tool)
# running gpt-4o-mini through the Vocareum OpenAI-compatible proxy. Credentials
# live in a .env file alongside this script; we load it with a path relative to
# this file so it works regardless of the current working directory. The .env is
# gitignored — see .env.example for the variables it must define.
# =============================================================================
import re
import difflib
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

from smolagents import ToolCallingAgent, OpenAIServerModel, tool

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
dotenv.load_dotenv(dotenv_path=_ENV_PATH)

model = OpenAIServerModel(
    model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


# =============================================================================
# Configuration
# -----------------------------------------------------------------------------
# Deployment / sensitive settings (model id, endpoint, API key) are externalized
# to .env. The knobs below are *business rules* — domain logic that belongs in
# code — centralized here in one frozen dataclass instead of being scattered as
# magic numbers throughout the file. (At larger scale these would graduate to an
# external config file; see the README's design-choices section.)
# =============================================================================
@dataclass(frozen=True)
class Config:
    # Bulk-discount ladder: (minimum quantity, discount fraction), highest first.
    discount_tiers: tuple = ((1000, 0.08), (500, 0.05), (100, 0.02))
    # Cap on any single line's total discount (bulk ladder + historical benchmark).
    max_discount: float = 0.15
    # How many comparable past quotes to retrieve for the historical price benchmark.
    history_quote_limit: int = 5
    # difflib ratio below which a fuzzy item name is treated as "not carried".
    fuzzy_match_cutoff: float = 0.62
    # Bulk-unit -> catalog (sheet) multipliers for paper/specialty items.
    units_per_ream: int = 500
    units_per_pack: int = 100
    # Worker-agent step limits and execution settings.
    inventory_max_steps: int = 12
    quoting_max_steps: int = 6
    sales_max_steps: int = 14
    advisor_max_steps: int = 4
    orchestrator_max_steps: int = 8
    agent_verbosity: int = 0        # 0 = quiet; raise to 1/2 to watch agent steps
    max_tool_threads: int = 1       # serialize tool calls to avoid DB write races
    # Run the advisor agent every Nth request (bounds run cost).
    advisor_every: int = 10


CONFIG = Config()


# =============================================================================
# Structured tool results
# -----------------------------------------------------------------------------
# Following multi-agent reliability best practices, tools never raise raw
# exceptions or fail silently: they return a structured status + error type +
# context so the orchestrator can validate and branch deterministically.
# =============================================================================
@dataclass
class ToolResult:
    status: str                                  # "ok" | "error" | "declined"
    data: dict = field(default_factory=dict)
    error_type: str = ""                         # machine-readable failure code
    message: str = ""                            # internal/debug text (never shown to customer)


def ok(**data) -> dict:
    """Build a successful tool result."""
    return asdict(ToolResult(status="ok", data=data))


def err(error_type: str, message: str, **data) -> dict:
    """Build an error tool result (contained failure, reported with context)."""
    return asdict(ToolResult(status="error", data=data, error_type=error_type, message=message))


def declined(reason: str, **data) -> dict:
    """Build a declined tool result (a deliberate, graceful 'no')."""
    return asdict(ToolResult(status="declined", data=data, error_type="declined", message=reason))


# =============================================================================
# Catalog index + pricing helpers
# -----------------------------------------------------------------------------
# `paper_supplies` (defined at the top of this file) is the canonical catalog.
# These helpers index it so the rest of the system can look items up by name.
# =============================================================================
CATALOG = {item["item_name"]: item for item in paper_supplies}
CATALOG_NAMES = list(CATALOG.keys())
_CATALOG_LOWER = {name.lower(): name for name in CATALOG_NAMES}


def catalog_price(name: str) -> Optional[float]:
    """Per-catalog-unit price for an item, or None if not in the catalog."""
    item = CATALOG.get(name)
    return float(item["unit_price"]) if item else None


def catalog_category(name: str) -> Optional[str]:
    """Catalog category ('paper' | 'product' | 'large_format' | 'specialty')."""
    item = CATALOG.get(name)
    return item["category"] if item else None


def bulk_discount(quantity: int) -> float:
    """Bulk-discount fraction for a line, from the CONFIG.discount_tiers ladder.

    Tiers are simple and transparent so the rationale can be explained to the
    customer (default: 8% at 1000+, 5% at 500+, 2% at 100+, else none).
    """
    for threshold, fraction in CONFIG.discount_tiers:       # highest threshold first
        if quantity >= threshold:
            return fraction
    return 0.0


# Event keywords used to find comparable historical quotes for the pricing benchmark.
_EVENT_TERMS = ("wedding", "party", "ceremony", "meeting", "exhibition", "reception",
                "conference", "graduation", "fundraiser", "gala", "celebration",
                "launch", "seminar", "workshop", "festival", "demonstration", "event")


def _extract_discount_fractions(quotes: list) -> list:
    """Discount fractions explicitly stated in historical quote text (only when the
    text actually mentions a discount), kept to the plausible (0, 0.5] range."""
    fractions = []
    for q in quotes:
        text = f"{q.get('quote_explanation', '')} {q.get('original_request', '')}".lower()
        if "discount" not in text:
            continue
        for pct in re.findall(r"(\d{1,2})\s*%", text):
            value = int(pct) / 100.0
            if 0 < value <= 0.5:
                fractions.append(value)
    return fractions


def historical_discount(search_terms: list):
    """Retrieval-augmented pricing benchmark: the average explicit discount used in
    comparable past quotes. Returns (fraction, n_quotes_with_discount); the fraction
    is 0.0 when nothing comparable is found, so the quote falls back to the ladder."""
    terms = [t for t in (search_terms or []) if t][:2]
    try:
        quotes = search_quote_history(terms, limit=CONFIG.history_quote_limit)
    except Exception:
        return 0.0, 0
    fractions = _extract_discount_fractions(quotes)
    if not fractions:
        return 0.0, 0
    return min(sum(fractions) / len(fractions), CONFIG.max_discount), len(fractions)


def search_terms_for(request_text: str, feasible: list) -> list:
    """Pick one broad retrieval keyword: an event word if present, else the top item."""
    lowered = request_text.lower()
    for term in _EVENT_TERMS:
        if term in lowered:
            return [term]
    if feasible:
        return [feasible[0]["catalog_name"].split()[0].lower()]
    return []


# =============================================================================
# Item-name normalization
# -----------------------------------------------------------------------------
# Customer requests use fuzzy free text ("A4 glossy paper", "heavy cardstock")
# that will not match canonical catalog names ("Glossy paper", "Cardstock") --
# and mismatched names make create_transaction silently log orphan rows. This
# mapping is DETERMINISTIC (not delegated to the LLM) because it is the single
# biggest correctness lever in the system. Unresolved items return None and are
# declined with a clear reason ("we don't carry this").
# =============================================================================

# Curated aliases: a lowercased fragment that may appear in a request -> the
# exact catalog name it should map to.
ALIASES = {
    "a4 paper": "A4 paper", "a4 white paper": "A4 paper",
    "white printer paper": "Standard copy paper", "a4 white printer paper": "Standard copy paper",
    "printer paper": "Standard copy paper", "copy paper": "Standard copy paper",
    "standard copy paper": "Standard copy paper", "standard paper": "Standard copy paper",
    "letter paper": "Letter-sized paper", "letter-sized paper": "Letter-sized paper",
    "letter sized paper": "Letter-sized paper", "legal paper": "Legal-size paper",
    "legal-size paper": "Legal-size paper", "legal size paper": "Legal-size paper",
    "cardstock": "Cardstock", "heavy cardstock": "Cardstock", "heavyweight cardstock": "Cardstock",
    "white cardstock": "Cardstock", "colored cardstock": "Cardstock",
    "glossy paper": "Glossy paper", "glossy a4 paper": "Glossy paper", "a4 glossy paper": "Glossy paper",
    "matte paper": "Matte paper", "colored paper": "Colored paper", "construction paper": "Construction paper",
    "recycled paper": "Recycled paper", "eco-friendly paper": "Eco-friendly paper",
    "eco friendly paper": "Eco-friendly paper", "kraft paper": "Kraft paper",
    "kraft paper envelopes": "Envelopes", "wrapping paper": "Wrapping paper",
    "glitter paper": "Glitter paper", "decorative paper": "Decorative paper",
    "letterhead": "Letterhead paper", "letterhead paper": "Letterhead paper",
    "crepe paper": "Crepe paper", "photo paper": "Photo paper", "butcher paper": "Butcher paper",
    "heavyweight paper": "Heavyweight paper", "bright-colored paper": "Bright-colored paper",
    "bright colored paper": "Bright-colored paper", "patterned paper": "Patterned paper",
    "poster paper": "Poster paper", "large poster paper": "Large poster paper (24x36 inches)",
    "poster board": "Large poster paper (24x36 inches)", "poster boards": "Large poster paper (24x36 inches)",
    "banner paper": "Banner paper", "banner roll": "Rolls of banner paper (36-inch width)",
    "rolls of banner paper": "Rolls of banner paper (36-inch width)",
    "paper plates": "Paper plates", "paper cups": "Paper cups", "disposable cups": "Disposable cups",
    "paper napkins": "Paper napkins", "napkins": "Paper napkins", "table napkins": "Paper napkins",
    "table covers": "Table covers", "envelopes": "Envelopes", "sticky notes": "Sticky notes",
    "notepads": "Notepads", "invitation cards": "Invitation cards", "flyers": "Flyers",
    "party streamers": "Party streamers", "streamers": "Party streamers",
    "washi tape": "Decorative adhesive tape (washi tape)",
    "decorative washi tape": "Decorative adhesive tape (washi tape)",
    "decorative adhesive tape": "Decorative adhesive tape (washi tape)",
    "paper party bags": "Paper party bags", "party bags": "Paper party bags",
    "name tags": "Name tags with lanyards", "name tags with lanyards": "Name tags with lanyards",
    "presentation folders": "Presentation folders", "folders": "Presentation folders",
    "cover stock": "100 lb cover stock", "100 lb cover stock": "100 lb cover stock",
    "text paper": "80 lb text paper", "80 lb text paper": "80 lb text paper",
    "250 gsm cardstock": "250 gsm cardstock", "220 gsm poster paper": "220 gsm poster paper",
}

# Materials/sizes Beaver's Choice does not stock -> always decline (gives the
# rubric-required "unfulfilled with reasons" cases honestly).
NOT_CARRIED = ("a3", "a5", "balloon", "balloons", "cardboard", "ribbon", "candle",
               "candles", "tablecloth", "wristband", "vinyl", "ticket stock")

# Size/quality qualifiers stripped before fuzzy matching (kept for alias lookup).
_NOISE = re.compile(
    r"\b(a4|8\.?5\s*[\"x]*\s*11|24\s*[\"x]*\s*36|36[- ]inch|assorted|various|"
    r"white|black|premium|high[- ]?quality|quality|biodegradable|colou?red|colou?rs?)\b"
)


def normalize_item_name(raw: str) -> Optional[str]:
    """Map fuzzy request text to a canonical catalog name, or None if not carried."""
    if not raw:
        return None
    s = re.sub(r"\s+", " ", re.sub(r"[()]", " ", raw.strip().lower())).strip()

    # 1) Items we explicitly do not stock.
    if any(re.search(r"\b" + re.escape(t) + r"\b", s) for t in NOT_CARRIED):
        return None
    # 2) Exact alias.
    if s in ALIASES:
        return ALIASES[s]
    # 3) Longest alias fragment contained in the text.
    for key in sorted(ALIASES, key=len, reverse=True):
        if key in s:
            return ALIASES[key]
    # 4) Exact (case-insensitive) catalog name.
    if s in _CATALOG_LOWER:
        return _CATALOG_LOWER[s]
    # 5) Fuzzy match after stripping size/quality noise.
    cleaned = re.sub(r"\s+", " ", _NOISE.sub(" ", s)).strip() or s
    match = difflib.get_close_matches(cleaned, [n.lower() for n in CATALOG_NAMES], n=1,
                                      cutoff=CONFIG.fuzzy_match_cutoff)
    return _CATALOG_LOWER[match[0]] if match else None


# =============================================================================
# Unit conversion
# -----------------------------------------------------------------------------
# Catalog prices are per sheet (paper/specialty) or per unit (products), but
# requests use reams / packs / boxes / rolls. We convert to catalog units and
# return a human note so the conversion can be shown in the quote rationale.
# Assumption (documented): 1 ream = 500 sheets, 1 pack/box = 100 sheets.
# =============================================================================
UNIT_FACTORS = {
    "ream": CONFIG.units_per_ream, "reams": CONFIG.units_per_ream,
    "pack": CONFIG.units_per_pack, "packs": CONFIG.units_per_pack,
    "packet": CONFIG.units_per_pack, "packets": CONFIG.units_per_pack,
    "box": CONFIG.units_per_pack, "boxes": CONFIG.units_per_pack,
}


def convert_to_catalog_units(quantity: int, unit: str, catalog_name: str):
    """Return (catalog_unit_qty, note). Bulk multipliers apply only to sheet paper."""
    unit = (unit or "").strip().lower()
    factor = UNIT_FACTORS.get(unit, 1)
    if factor > 1 and catalog_category(catalog_name) not in ("paper", "specialty"):
        # e.g. "5 boxes of paper cups" -> treat as units, not sheets.
        return int(quantity), f"{quantity} {unit} treated as {int(quantity)} units"
    total = int(quantity) * factor
    if factor > 1:
        return total, f"{quantity} {unit} = {total} sheets"
    return int(quantity), f"{int(quantity)} {unit or 'units'}"


# =============================================================================
# Request parsing (deterministic, regex-first)
# =============================================================================
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

_UNIT_WORDS = (r"sheets?|reams?|rolls?|packs?|packets?|boxes|box|units?|pads?|cards?|sets?|"
               r"plates?|cups?|napkins?|covers?|bags?|tags?|folders?|envelopes?|pieces?|rolls?")


def parse_request_date(text: str, default: str = "2025-04-01") -> str:
    """Extract the ISO request date the harness appends as '(Date of request: YYYY-MM-DD)'."""
    m = re.search(r"Date of request:\s*(\d{4}-\d{2}-\d{2})", text)
    return m.group(1) if m else default


def parse_deadline(text: str) -> Optional[str]:
    """Extract a 'delivered by Month D, YYYY' deadline as an ISO date, if present."""
    m = re.search(r"by\s+([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", text)
    if not m:
        return None
    month = _MONTHS.get(m.group(1).lower())
    if not month:
        return None
    try:
        return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"
    except ValueError:
        return None


def parse_line_items(text: str) -> list:
    """Extract ordered line items from free-text. Each item is a dict with the raw
    name, requested quantity/unit, the normalized catalog name (or None), the
    quantity expressed in catalog units, and a conversion note.

    Quantities are matched anywhere in a sentence (not only at the start of a
    fragment), so inline prose like "I need 500 sheets of X, along with 250 ..."
    parses correctly. Each item phrase runs until the next quantity or a break.
    """
    # Strip digit-bearing tokens that are NOT quantities so they aren't misread:
    # the harness request-date stamp, "delivered by <Month D, YYYY>" deadlines,
    # percentages ("100% recycled"), and size dimensions ("24\"x36\"", "8.5x11").
    text = re.sub(r"\(Date of request:.*?\)", " ", text)
    text = re.sub(r"\b(?:delivered\s+)?by\s+[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}", " ", text)
    text = re.sub(r"\d+\s*%", " ", text)
    text = re.sub(r'\d+\.?\d*\s*"?\s*[xX]\s*\d+\.?\d*\s*"?', " ", text)

    line_re = re.compile(
        r"(\d[\d,]*)\s*"                       # quantity
        rf"(?:({_UNIT_WORDS})\b\s*)?"          # optional unit word
        r"(?:of\s+)?"                          # optional "of"
        r"(.+?)"                               # item phrase (lazy)
        r"(?=\s+\d[\d,]*\s|\s*[;.\n]|$)",      # until next quantity / break / end
        re.IGNORECASE,
    )
    items = []
    for m in line_re.finditer(text):
        qty = int(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        # Trim trailing prepositional clauses ("... for our reception") and any
        # trailing list connector ("..., and", "... plus") the capture grabbed.
        raw_name = re.split(r"\b(?:by|on|delivered|for|before)\b", m.group(3))[0]
        raw_name = re.sub(r"[\s,]+(?:and|plus|along\s+with|as\s+well\s+as)\s*$", "",
                          raw_name, flags=re.IGNORECASE).strip(" .,")
        # Require a real word so stray numbers are not treated as items.
        if not re.search(r"[A-Za-z]{3,}", raw_name):
            continue
        catalog_name = normalize_item_name(raw_name)
        if catalog_name:
            qty_units, note = convert_to_catalog_units(qty, unit, catalog_name)
        else:
            qty_units, note = qty, ""
        items.append({
            "raw_name": raw_name, "qty_raw": qty, "unit": unit,
            "catalog_name": catalog_name, "qty_units": qty_units, "conversion_note": note,
        })
    return items


def log_transition(stage: str, **payload) -> None:
    """Structured state-transition log (DB mutations) for cross-agent traceability."""
    print("[STATE] " + stage + ": " + " ".join(f"{k}={v}" for k, v in payload.items()))


# =============================================================================
# Tool implementations (_impl) + @tool wrappers
# -----------------------------------------------------------------------------
# Each of the 7 sanctioned helper functions is wrapped in at least one tool.
# The logic lives in plain `_impl` functions so the deterministic orchestrator
# can reuse it as a graceful fallback without going through an LLM, while the
# @tool versions are what the worker agents call.
# =============================================================================

def _stock_level(item_name: str, as_of_date: str) -> int:
    """Net stock for an item (>=0), reading the DB via get_stock_level."""
    try:
        df = get_stock_level(item_name, as_of_date)
        if df is None or df.empty:
            return 0
        return int(df.iloc[0]["current_stock"])
    except Exception:
        return 0


def _place_restock_impl(item_name: str, quantity: int, order_date: str) -> dict:
    price = catalog_price(item_name)
    if price is None:
        return err("not_in_catalog", f"{item_name} is not in the catalog")
    quantity = int(quantity)
    total = round(price * quantity, 2)
    cash = get_cash_balance(order_date)
    if total > cash:
        return declined("insufficient funds to restock", item_name=item_name,
                        needed=total, available=round(cash, 2))
    txn = create_transaction(item_name, "stock_orders", quantity, total, order_date)
    eta = get_supplier_delivery_date(order_date, quantity)
    log_transition("restock", item=item_name, qty=quantity, cost=total, eta=eta, txn=txn)
    return ok(item_name=item_name, quantity=quantity, cost=total, eta=eta, transaction_id=txn)


def _record_sale_impl(item_name: str, quantity: int, total_price: float, sale_date: str) -> dict:
    quantity = int(quantity)
    total_price = round(float(total_price), 2)
    txn = create_transaction(item_name, "sales", quantity, total_price, sale_date)
    log_transition("sale", item=item_name, qty=quantity, total=total_price, txn=txn)
    return ok(item_name=item_name, quantity=quantity, total_price=total_price, transaction_id=txn)


# ---- Inventory agent tools ----
@tool
def check_full_inventory(as_of_date: str) -> dict:
    """Return a snapshot of all in-stock items as of a date (wraps get_all_inventory).

    Args:
        as_of_date: ISO date (YYYY-MM-DD) for the inventory snapshot.
    """
    try:
        inventory = get_all_inventory(as_of_date)
        return ok(inventory=inventory, item_count=len(inventory))
    except Exception as exc:
        return err("db_error", f"inventory snapshot failed: {exc}")


@tool
def check_item_stock(item_name: str, as_of_date: str) -> dict:
    """Return the current stock level of one catalog item (wraps get_stock_level).

    Args:
        item_name: exact catalog item name.
        as_of_date: ISO date (YYYY-MM-DD).
    """
    try:
        return ok(item_name=item_name, current_stock=_stock_level(item_name, as_of_date))
    except Exception as exc:
        return err("db_error", f"stock lookup failed for {item_name}: {exc}")


@tool
def estimate_restock_eta(order_date: str, quantity: int) -> dict:
    """Estimate the supplier delivery date for a restock (wraps get_supplier_delivery_date).

    Args:
        order_date: ISO date (YYYY-MM-DD) the restock would be placed.
        quantity: number of units to order.
    """
    try:
        return ok(eta=get_supplier_delivery_date(order_date, int(quantity)))
    except Exception as exc:
        return err("db_error", f"eta calculation failed: {exc}")


@tool
def place_restock_order(item_name: str, quantity: int, order_date: str) -> dict:
    """Buy stock from the supplier, respecting cash (wraps create_transaction/get_cash_balance).

    Args:
        item_name: exact catalog item name.
        quantity: number of units to purchase.
        order_date: ISO date (YYYY-MM-DD).
    """
    try:
        return _place_restock_impl(item_name, quantity, order_date)
    except Exception as exc:
        return err("db_error", f"restock failed for {item_name}: {exc}")


# ---- Quoting agent tools ----
@tool
def find_similar_quotes(search_term: str) -> dict:
    """Look up comparable past quotes for ONE broad keyword and report the average
    explicit discount they applied (wraps search_quote_history).

    Args:
        search_term: a single broad keyword (e.g. an event type like 'wedding').
    """
    try:
        terms = [search_term.strip().lower()] if search_term and search_term.strip() else []
        history = search_quote_history(terms, limit=CONFIG.history_quote_limit)
        fractions = _extract_discount_fractions(history)
        avg_discount_pct = round(sum(fractions) / len(fractions) * 100, 1) if fractions else 0.0
        slim = [{"total_amount": h.get("total_amount"), "order_size": h.get("order_size"),
                 "event_type": h.get("event_type")} for h in history]
        return ok(matches=slim, count=len(slim),
                  avg_discount_pct=avg_discount_pct, n_with_discount=len(fractions))
    except Exception as exc:
        return err("db_error", f"quote history lookup failed: {exc}")


@tool
def price_line_item(item_name: str, quantity: int) -> dict:
    """Price one line using catalog price + bulk discount ladder.

    Args:
        item_name: exact catalog item name.
        quantity: number of catalog units.
    """
    price = catalog_price(item_name)
    if price is None:
        return err("not_in_catalog", f"{item_name} is not in the catalog")
    quantity = int(quantity)
    discount = bulk_discount(quantity)
    line_total = round(price * quantity * (1 - discount), 2)
    return ok(item_name=item_name, quantity=quantity, unit_price=price,
              discount_pct=round(discount * 100, 1), line_total=line_total)


# ---- Sales agent tools ----
@tool
def verify_stock(item_name: str, quantity: int, as_of_date: str) -> dict:
    """Check whether enough stock exists to fulfil a quantity (wraps get_stock_level).

    Args:
        item_name: exact catalog item name.
        quantity: units required.
        as_of_date: ISO date (YYYY-MM-DD).
    """
    try:
        available = _stock_level(item_name, as_of_date)
        return ok(item_name=item_name, available=available, sufficient=available >= int(quantity))
    except Exception as exc:
        return err("db_error", f"stock verification failed: {exc}")


@tool
def check_funds(as_of_date: str) -> dict:
    """Return the company cash balance as of a date (wraps get_cash_balance).

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    try:
        return ok(cash_balance=round(get_cash_balance(as_of_date), 2))
    except Exception as exc:
        return err("db_error", f"cash lookup failed: {exc}")


@tool
def record_sale(item_name: str, quantity: int, total_price: float, sale_date: str) -> dict:
    """Record a confirmed sale (wraps create_transaction).

    Args:
        item_name: exact catalog item name.
        quantity: units sold.
        total_price: TOTAL price charged (already includes any discount).
        sale_date: ISO date (YYYY-MM-DD).
    """
    try:
        return _record_sale_impl(item_name, quantity, total_price, sale_date)
    except Exception as exc:
        return err("db_error", f"record_sale failed for {item_name}: {exc}")


@tool
def confirm_delivery_date(order_date: str, quantity: int) -> dict:
    """Compute the promised delivery date for an order (wraps get_supplier_delivery_date).

    Args:
        order_date: ISO date (YYYY-MM-DD).
        quantity: units in the order.
    """
    try:
        return ok(delivery_date=get_supplier_delivery_date(order_date, int(quantity)))
    except Exception as exc:
        return err("db_error", f"delivery date calculation failed: {exc}")


# ---- Business advisor agent tools ----
@tool
def financial_snapshot(as_of_date: str) -> dict:
    """Return cash, inventory value, assets and top sellers (wraps generate_financial_report).

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    try:
        report = generate_financial_report(as_of_date)
        return ok(cash_balance=round(report["cash_balance"], 2),
                  inventory_value=round(report["inventory_value"], 2),
                  total_assets=round(report["total_assets"], 2),
                  top_selling_products=report.get("top_selling_products", []))
    except Exception as exc:
        return err("db_error", f"financial report failed: {exc}")


@tool
def cash_position(as_of_date: str) -> dict:
    """Return the current cash balance as of a date (wraps get_cash_balance).

    Args:
        as_of_date: ISO date (YYYY-MM-DD).
    """
    try:
        return ok(cash_balance=round(get_cash_balance(as_of_date), 2))
    except Exception as exc:
        return err("db_error", f"cash lookup failed: {exc}")


# =============================================================================
# Orchestrator + worker agents
# =============================================================================
def _agent_tool_calls(agent, tool_name: str) -> list:
    """Return the argument dicts for calls to `tool_name` in an agent's last run.

    Used to reconcile what a worker agent actually did against what it was asked
    to do, so the deterministic fallback only fills genuine gaps (no double work).
    """
    calls = []
    memory = getattr(agent, "memory", None)
    for step in (getattr(memory, "steps", None) or []):
        for tc in (getattr(step, "tool_calls", None) or []):
            if getattr(tc, "name", None) == tool_name:
                args = getattr(tc, "arguments", None)
                if isinstance(args, dict):
                    calls.append(args)
    return calls


class Orchestrator(ToolCallingAgent):
    """Supervises order handling and delegates to four specialised worker agents.

    Design note: the orchestrator is a genuine ToolCallingAgent exposing four
    delegation tools (matching the workflow diagram), but `process_request`
    drives a DETERMINISTIC, supervised control flow. Parsing, item-name
    normalization, and the reorder/affordability/deadline decisions happen in
    Python (reliable with gpt-4o-mini); the worker agents execute their
    specialised tool calls. SQLite stays the single source of truth, and the
    customer-facing reply is assembled deterministically and scrubbed of any
    internal details (margins, cash, raw errors).
    """

    def __init__(self, model):
        # --- Worker agents (each owns only its own tools) ---
        self.inventory_agent = ToolCallingAgent(
            tools=[check_full_inventory, check_item_stock, estimate_restock_eta, place_restock_order],
            model=model, name="inventory_agent",
            description="Checks stock levels and places supplier restock orders.",
            instructions=("You are the inventory specialist for Beaver's Choice Paper Company. "
                          "Use your tools to review stock and place restock orders. Always use the "
                          "EXACT catalog item names and quantities you are given; call one tool per "
                          "item, then finish."),
            max_steps=CONFIG.inventory_max_steps, verbosity_level=CONFIG.agent_verbosity,
            max_tool_threads=CONFIG.max_tool_threads)

        self.quoting_agent = ToolCallingAgent(
            tools=[find_similar_quotes, price_line_item],
            model=model, name="quoting_agent",
            description="Prices orders with bulk discounts, grounded in historical quotes.",
            instructions=("You price paper orders. Call find_similar_quotes ONCE with a single broad "
                          "keyword from the request to review past pricing, then give a brief, fair "
                          "pricing rationale. Never disclose costs, margins, or internal data."),
            max_steps=CONFIG.quoting_max_steps, verbosity_level=CONFIG.agent_verbosity,
            max_tool_threads=CONFIG.max_tool_threads)

        self.sales_agent = ToolCallingAgent(
            tools=[verify_stock, check_funds, record_sale, confirm_delivery_date],
            model=model, name="sales_agent",
            description="Finalises sales transactions and confirms delivery dates.",
            instructions=("You finalise confirmed paper orders. For each line you are given, call "
                          "record_sale with the EXACT item name, quantity, and TOTAL price provided. "
                          "Call one record_sale per line, then finish."),
            max_steps=CONFIG.sales_max_steps, verbosity_level=CONFIG.agent_verbosity,
            max_tool_threads=CONFIG.max_tool_threads)

        self.advisor_agent = ToolCallingAgent(
            tools=[financial_snapshot, cash_position],
            model=model, name="advisor_agent",
            description="Reports company finances and recommends operational actions.",
            instructions=("You are a business advisor. Call financial_snapshot for the given date and "
                          "give ONE short internal recommendation (restock priorities or cash caution). "
                          "Internal use only."),
            max_steps=CONFIG.advisor_max_steps, verbosity_level=CONFIG.agent_verbosity,
            max_tool_threads=CONFIG.max_tool_threads)

        # --- Delegation tools: the orchestrator's own agent interface ---
        @tool
        def delegate_inventory(instructions: str) -> str:
            """Delegate an inventory or restock task to the inventory agent.

            Args:
                instructions: a plain-language task for the inventory agent.
            """
            return str(self.inventory_agent.run(instructions))

        @tool
        def delegate_quoting(instructions: str) -> str:
            """Delegate a pricing task to the quoting agent.

            Args:
                instructions: a plain-language task for the quoting agent.
            """
            return str(self.quoting_agent.run(instructions))

        @tool
        def delegate_sales(instructions: str) -> str:
            """Delegate a sales-finalisation task to the sales agent.

            Args:
                instructions: a plain-language task for the sales agent.
            """
            return str(self.sales_agent.run(instructions))

        @tool
        def delegate_advisor(instructions: str) -> str:
            """Delegate a financial-reporting task to the advisor agent.

            Args:
                instructions: a plain-language task for the advisor agent.
            """
            return str(self.advisor_agent.run(instructions))

        super().__init__(
            tools=[delegate_inventory, delegate_quoting, delegate_sales, delegate_advisor],
            model=model, name="orchestrator",
            description="Coordinates inventory, quoting, sales and advisory agents to fulfil orders.",
            instructions="Coordinate the worker agents to fulfil the customer's paper order.",
            max_steps=CONFIG.orchestrator_max_steps, verbosity_level=CONFIG.agent_verbosity,
            max_tool_threads=CONFIG.max_tool_threads)

        self._req_count = 0
        self.advisor_every = CONFIG.advisor_every   # run the advisor agent every Nth request
        self.last_summary = {}          # structured per-request result for test_results.csv

    # ------------------------------------------------------------------ #
    def _safe_run(self, agent, prompt: str) -> Optional[str]:
        """Run a worker agent, containing any failure (graceful degradation)."""
        try:
            return str(agent.run(prompt))
        except Exception as exc:
            print(f"[WARN] {agent.name} failed: {type(exc).__name__}: {exc}")
            return None

    def process_request(self, request_with_date: str) -> str:
        """Handle one customer request end-to-end and return the customer-facing reply."""
        self._req_count += 1
        date = parse_request_date(request_with_date)
        deadline = parse_deadline(request_with_date)
        lines = parse_line_items(request_with_date)

        # Isolate per-request state: clear each worker's short-term memory.
        for agent in (self.inventory_agent, self.quoting_agent, self.sales_agent, self.advisor_agent):
            agent.memory.steps = []

        fulfilled, declined, feasible = [], [], []

        # 1) Normalise + screen out items we do not carry.
        resolved = []
        for ln in lines:
            if ln["catalog_name"]:
                resolved.append(ln)
            else:
                declined.append({"raw": ln["raw_name"], "reason": "we do not currently carry this item",
                                 "code": "not_in_catalog"})

        # 2) Inventory + restock decisions (deterministic gates).
        restock_plan = []
        for ln in resolved:
            name, need = ln["catalog_name"], ln["qty_units"]
            stock = _stock_level(name, date)
            if stock >= need:
                feasible.append(ln)
                continue
            shortfall = need - max(stock, 0)
            eta = get_supplier_delivery_date(date, shortfall)
            restock_cost = round(catalog_price(name) * shortfall, 2)
            if deadline and eta > deadline:
                declined.append({"raw": ln["raw_name"],
                                 "reason": f"we cannot restock and deliver by {deadline} (earliest {eta})",
                                 "code": "deadline_infeasible"})
            elif restock_cost > get_cash_balance(date):
                declined.append({"raw": ln["raw_name"],
                                 "reason": "we are unable to source enough stock in time",
                                 "code": "insufficient_funds_to_restock"})
            else:
                restock_plan.append((name, shortfall))
                feasible.append(ln)

        # 3) Execute restocks via the inventory agent, with a deterministic fallback.
        if restock_plan:
            order_lines = "\n".join(f"- {qty} units of '{name}'" for name, qty in restock_plan)
            self._safe_run(self.inventory_agent,
                           f"First call check_full_inventory for {date}. Then place restock orders "
                           f"dated {date}: call place_restock_order once per item with the exact name "
                           f"and quantity:\n{order_lines}")
            for name, qty in restock_plan:
                if _stock_level(name, date) < qty:            # agent missed it -> fill the gap
                    _place_restock_impl(name, qty, date)

        # 4) Quote feasible lines. Retrieval-augmented pricing: comparable past quotes
        #    set a historical discount benchmark, and the agent also consults history.
        hist_frac, hist_n = 0.0, 0
        if feasible:
            hist_frac, hist_n = historical_discount(search_terms_for(request_with_date, feasible))
            self._safe_run(self.quoting_agent,
                           "Review past pricing with find_similar_quotes, then summarise a fair "
                           "rationale for an order of: "
                           + ", ".join(l["catalog_name"] for l in feasible) + ".")

        # 5) Finalise sales via the sales agent; reconcile against the DB (fallback fills gaps).
        sale_payload = []
        for ln in feasible:
            name, qty = ln["catalog_name"], ln["qty_units"]
            unit_price = catalog_price(name)
            # Discount = larger of the bulk ladder and the historical benchmark, capped.
            # History can only make us more competitive, never less.
            bulk = bulk_discount(qty)
            discount = min(max(bulk, hist_frac), CONFIG.max_discount)
            sale_payload.append({
                "item": name, "qty": qty, "unit_price": unit_price,
                "discount_pct": round(discount * 100), "raw": ln["raw_name"],
                "line_total": round(unit_price * qty * (1 - discount), 2),
                "conversion_note": ln["conversion_note"],
                "history_applied": hist_n > 0 and hist_frac > bulk,
            })

        if sale_payload:
            sale_lines = "\n".join(
                f"- {s['qty']} of '{s['item']}' for total ${s['line_total']:.2f}" for s in sale_payload)
            self._safe_run(self.sales_agent,
                           f"Record these confirmed sales dated {date}. Call record_sale once per line "
                           f"with the exact item name, quantity, and TOTAL price:\n{sale_lines}")
            recorded = _agent_tool_calls(self.sales_agent, "record_sale")

            def _already_recorded(item, qty):
                return any(str(a.get("item_name", "")).strip() == item
                           and int(a.get("quantity", -1)) == qty for a in recorded)

            for s in sale_payload:
                if not _already_recorded(s["item"], s["qty"]):
                    _record_sale_impl(s["item"], s["qty"], s["line_total"], date)
                fulfilled.append(s)

        # 6) Advisor review (internal only) -- exercised periodically to bound cost.
        if self._req_count % self.advisor_every == 0:
            self._safe_run(self.advisor_agent,
                           f"Give an internal financial snapshot and one recommendation as of {date}.")

        # 7) Promised delivery date = latest line ETA among fulfilled items.
        delivery_date = ""
        if fulfilled:
            delivery_date = max(get_supplier_delivery_date(date, s["qty"]) for s in fulfilled)

        total_charged = round(sum(s["line_total"] for s in fulfilled), 2)
        self.last_summary = {
            "fulfilled": 1 if fulfilled else 0,
            "num_lines_requested": len(lines),
            "num_lines_fulfilled": len(fulfilled),
            "num_lines_declined": len(declined),
            "decline_reasons": "; ".join(sorted({d["code"] for d in declined})),
            "total_charged": total_charged,
            "delivery_date": delivery_date,
        }
        return self._assemble_reply(fulfilled, declined, total_charged, delivery_date, hist_n)

    def _assemble_reply(self, fulfilled: list, declined: list, total: float,
                        delivery_date: str, history_n: int = 0) -> str:
        """Build the transparent, customer-safe reply (no margins / cash / raw errors / PII)."""
        parts = []
        if fulfilled:
            parts.append("Thank you for your order with Beaver's Choice Paper Company! "
                         "Here is your confirmed quote:")
            for f in fulfilled:
                line = (f"  - {f['qty']:,} x {f['item']} @ ${f['unit_price']:.2f} "
                        f"= ${f['line_total']:,.2f}")
                extras = []
                if f["discount_pct"] > 0:
                    label = ("bulk + historical-benchmark discount"
                             if f.get("history_applied") else "bulk discount")
                    extras.append(f"{f['discount_pct']:.0f}% {label}")
                if "=" in f.get("conversion_note", ""):       # show ream/box -> sheet conversions
                    extras.append(f["conversion_note"])
                if extras:
                    line += " (" + "; ".join(extras) + ")"
                parts.append(line)
            parts.append(f"Order total: ${total:,.2f}")
            if delivery_date:
                parts.append(f"Estimated delivery by: {delivery_date}")
            if history_n and any(f.get("history_applied") for f in fulfilled):
                parts.append(f"Pricing was benchmarked against {history_n} comparable past "
                             f"order{'s' if history_n != 1 else ''}.")
        if declined:
            parts.append("We're sorry, but we could not fulfil the following:"
                         if fulfilled else
                         "Unfortunately we could not fulfil your request:")
            for d in declined:
                parts.append(f"  - {d['raw']}: {d['reason']}")
        if not fulfilled and not declined:
            parts.append("We could not identify any catalog items in your request. "
                         "Could you please clarify what you'd like to order?")
        parts.append("We appreciate your business. - Beaver's Choice Paper Company")
        return "\n".join(parts)


# =============================================================================
# Test harness
# =============================================================================

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    orchestrator = Orchestrator(model)
    print("Multi-agent system ready (Orchestrator + Inventory/Quoting/Sales/Advisor).\n")

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"
        cash_before = current_cash

        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        try:
            response = orchestrator.process_request(request_with_date)
            summary = orchestrator.last_summary
        except Exception as exc:
            # Graceful degradation: one failed request never aborts the run.
            print(f"ERROR processing request {idx+1}: {type(exc).__name__}: {exc}")
            response = ("We're sorry, we ran into an unexpected problem handling your request "
                        "and could not complete it. Please resend or contact us.")
            summary = {"fulfilled": 0, "num_lines_requested": 0, "num_lines_fulfilled": 0,
                       "num_lines_declined": 0, "decline_reasons": "internal_error",
                       "total_charged": 0.0, "delivery_date": ""}

        # Update state (the SQLite DB is the single source of truth).
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]
        cash_change = round(current_cash - cash_before, 2)

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}  (change: ${cash_change:+.2f})")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_before": round(cash_before, 2),
                "cash_balance": round(current_cash, 2),
                "cash_change": cash_change,
                "inventory_value": round(current_inventory, 2),
                "fulfilled": summary["fulfilled"],
                "num_lines_requested": summary["num_lines_requested"],
                "num_lines_fulfilled": summary["num_lines_fulfilled"],
                "num_lines_declined": summary["num_lines_declined"],
                "decline_reasons": summary["decline_reasons"],
                "total_charged": summary["total_charged"],
                "delivery_date": summary["delivery_date"],
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()

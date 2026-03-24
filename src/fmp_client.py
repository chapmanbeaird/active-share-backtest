"""
FMP (Financial Modeling Prep) API Client
Provides historical price data, dividends, and quality metrics with disk-based caching.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class FMPClient:
    """
    Client for Financial Modeling Prep API with disk-based caching.

    Features:
    - Automatic caching of API responses to disk
    - Rate limiting (300 calls/min for Starter plan)
    - Retry logic with exponential backoff
    - Support for historical prices, dividends, and quality metrics
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    STABLE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str = None, cache_dir: str = "./cache/fmp",
                 rate_limit: int = 300, use_cache: bool = True):
        """
        Initialize FMP client.

        Args:
            api_key: FMP API key. If None, loads from FMP_API_KEY env var.
            cache_dir: Directory for caching API responses.
            rate_limit: Max API calls per minute (default 300 for Starter plan).
            use_cache: Whether to use disk caching.
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key required. Set FMP_API_KEY in .env or pass api_key parameter.")

        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.rate_limit = rate_limit
        self._call_timestamps: List[float] = []

        # Create cache directories
        if self.use_cache:
            (self.cache_dir / "prices").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "dividends").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "profiles").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "scores").mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self._failed_symbols: set = set()  # Track symbols that return 402/404 to avoid retrying

    def _rate_limit_wait(self):
        """Enforce rate limiting by waiting if necessary."""
        now = time.time()
        # Remove timestamps older than 1 minute
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]

        if len(self._call_timestamps) >= self.rate_limit:
            # Wait until oldest call is more than 1 minute old
            oldest = min(self._call_timestamps)
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)

        self._call_timestamps.append(time.time())

    def _get_cache_path(self, category: str, key: str) -> Path:
        """Get cache file path for a given category and key."""
        # Create safe filename from key
        safe_key = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.cache_dir / category / f"{key.replace('/', '_')}_{safe_key}.json"

    def _load_from_cache(self, category: str, key: str) -> Optional[Any]:
        """Load data from cache if available."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(category, key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(self, category: str, key: str, data: Any):
        """Save data to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(category, key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except IOError as e:
            print(f"Warning: Could not save to cache: {e}")

    def _make_request(self, endpoint: str, params: Dict = None,
                      use_stable: bool = True, max_retries: int = 3) -> Any:
        """
        Make API request with rate limiting and retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_stable: Use stable API base URL
            max_retries: Maximum retry attempts

        Returns:
            JSON response data
        """
        # Skip symbols that have previously returned 402 (not available on FMP)
        symbol = params.get("symbol", "") if params else ""
        if symbol and symbol in self._failed_symbols:
            return None

        base_url = self.STABLE_URL if use_stable else self.BASE_URL
        url = f"{base_url}/{endpoint}"

        if params is None:
            params = {}
        params["apikey"] = self.api_key

        for attempt in range(max_retries):
            self._rate_limit_wait()

            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 402:  # Symbol not available on this plan
                    self._failed_symbols.add(symbol)
                    return None

                if response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt * 10  # Exponential backoff
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code >= 500:  # Server error
                    wait_time = 2 ** attempt
                    print(f"Server error {response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Error fetching {endpoint}: {e}")
                    return None
                time.sleep(2 ** attempt)

        return None

    def get_historical_prices(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Get historical end-of-day prices for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, adjClose, volume, dividend
        """
        cache_key = f"{symbol}_{start}_{end}"

        # Check cache
        cached = self._load_from_cache("prices", cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df

        # Fetch from API (stable endpoint)
        data = self._make_request(
            "historical-price-eod/full",
            params={"symbol": symbol, "from": start, "to": end},
            use_stable=True
        )

        if not data:
            return pd.DataFrame()

        # Handle different response formats
        if isinstance(data, dict) and 'historical' in data:
            records = data['historical']
        elif isinstance(data, list):
            records = data
        else:
            return pd.DataFrame()

        if not records:
            return pd.DataFrame()

        # Save to cache
        self._save_to_cache("prices", cache_key, records)

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')

    def get_dividends(self, symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Get dividend history for a symbol.

        Args:
            symbol: Ticker symbol
            start: Optional start date filter (YYYY-MM-DD)
            end: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, dividend, adjDividend, recordDate, paymentDate, declarationDate
        """
        cache_key = f"{symbol}_{start or 'all'}_{end or 'all'}"

        # Check cache
        cached = self._load_from_cache("dividends", cache_key)
        if cached is not None:
            df = pd.DataFrame(cached)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                return df

        # Fetch from API (stable endpoint)
        data = self._make_request("dividends", params={"symbol": symbol})

        if not data:
            return pd.DataFrame()

        # Handle response format (v3 API returns {"symbol": "X", "historical": [...]})
        if isinstance(data, dict) and 'historical' in data:
            records = data['historical']
        elif isinstance(data, list):
            records = data
        else:
            records = []

        if not records:
            return pd.DataFrame()

        # Save to cache
        self._save_to_cache("dividends", cache_key, records)

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])

        # Filter by date range if specified
        if start:
            df = df[df['date'] >= pd.to_datetime(start)]
        if end:
            df = df[df['date'] <= pd.to_datetime(end)]

        return df.sort_values('date')

    def get_company_profile(self, symbol: str) -> dict:
        """
        Get company profile information.

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with company profile data
        """
        cache_key = symbol

        cached = self._load_from_cache("profiles", cache_key)
        if cached is not None:
            return cached

        data = self._make_request("profile", params={"symbol": symbol})

        if not data:
            return {}

        # API returns list with single profile
        profile = data[0] if isinstance(data, list) and data else data

        if profile:
            self._save_to_cache("profiles", cache_key, profile)

        return profile or {}

    def get_financial_scores(self, symbol: str) -> dict:
        """
        Get financial scores (Piotroski F-Score, Altman Z-Score, etc.)

        Args:
            symbol: Ticker symbol

        Returns:
            Dictionary with financial scores
        """
        cache_key = symbol

        cached = self._load_from_cache("scores", cache_key)
        if cached is not None:
            return cached

        data = self._make_request("financial-scores", params={"symbol": symbol})

        if not data:
            return {}

        # API returns list with single score set
        scores = data[0] if isinstance(data, list) and data else data

        if scores:
            self._save_to_cache("scores", cache_key, scores)

        return scores or {}

    def get_total_return(self, symbol: str, start: str, end: str) -> tuple:
        """
        Calculate total return (price + dividends) for a symbol over a period.

        Args:
            symbol: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            Tuple of (total_return, price_return, dividend_return) as percentages,
            or (None, None, None) if data not available
        """
        # Get price data
        prices = self.get_historical_prices(symbol, start, end)
        if prices.empty or len(prices) < 2:
            return None, None, None

        # Ensure prices are sorted ascending by date (oldest first)
        prices = prices.sort_values('date', ascending=True).reset_index(drop=True)

        # Get start price (earliest date) and end price (latest date)
        close_col = 'close' if 'close' in prices.columns else 'adjClose'
        first_price = float(prices.iloc[0][close_col])
        last_price = float(prices.iloc[-1][close_col])

        if first_price is None or last_price is None or first_price == 0:
            return None, None, None

        price_return = ((last_price - first_price) / first_price) * 100

        # Get dividends and filter to date range
        dividends = self.get_dividends(symbol)  # Get all, then filter

        if dividends.empty:
            dividend_return = 0.0
        else:
            # Filter dividends to the date range
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            period_dividends = dividends[
                (dividends['date'] >= start_date) &
                (dividends['date'] <= end_date)
            ]

            # Sum dividends paid during period
            if period_dividends.empty or 'dividend' not in period_dividends.columns:
                dividend_return = 0.0
            else:
                total_dividends = period_dividends['dividend'].sum()
                dividend_return = (total_dividends / first_price) * 100

        total_return = price_return + dividend_return

        return total_return, price_return, dividend_return

    def get_batch_returns(self, symbols: List[str], start: str, end: str,
                          progress_callback=None) -> pd.DataFrame:
        """
        Get returns for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            DataFrame with columns: ticker, total_return, price_return, dividend_return
        """
        results = []
        missing = []

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(i + 1, len(symbols))

            total_ret, price_ret, div_ret = self.get_total_return(symbol, start, end)

            if total_ret is not None:
                results.append({
                    'ticker': symbol,
                    'total_return': total_ret,
                    'price_return': price_ret,
                    'dividend_return': div_ret
                })
            else:
                missing.append(symbol)

        df = pd.DataFrame(results)
        return df, missing

    def clear_cache(self, category: str = None):
        """
        Clear cached data.

        Args:
            category: Specific category to clear ('prices', 'dividends', etc.)
                     If None, clears all cache.
        """
        if category:
            cache_path = self.cache_dir / category
            if cache_path.exists():
                for f in cache_path.glob("*.json"):
                    f.unlink()
        else:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for f in subdir.glob("*.json"):
                        f.unlink()


# Example usage
if __name__ == "__main__":
    # Test the client
    client = FMPClient()

    # Test historical prices
    print("Testing historical prices...")
    prices = client.get_historical_prices("AAPL", "2023-01-01", "2023-12-31")
    print(f"Got {len(prices)} price records for AAPL")

    # Test dividends
    print("\nTesting dividends...")
    divs = client.get_dividends("AAPL", "2023-01-01", "2023-12-31")
    print(f"Got {len(divs)} dividend records for AAPL")

    # Test total return
    print("\nTesting total return calculation...")
    total, price, div = client.get_total_return("AAPL", "2023-01-01", "2023-12-31")
    if total is not None:
        print(f"AAPL 2023: Total={total:.2f}%, Price={price:.2f}%, Dividend={div:.2f}%")

    # Test company profile
    print("\nTesting company profile...")
    profile = client.get_company_profile("AAPL")
    print(f"Company: {profile.get('companyName', 'N/A')}, Sector: {profile.get('sector', 'N/A')}")

    # Test financial scores
    print("\nTesting financial scores...")
    scores = client.get_financial_scores("AAPL")
    print(f"Piotroski: {scores.get('piotroskiScore', 'N/A')}, Altman Z: {scores.get('altmanZScore', 'N/A')}")

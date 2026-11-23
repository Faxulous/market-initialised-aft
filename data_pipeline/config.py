"""
Configuration for downloading historical Premier League odds data from Betfair
Covers seasons 2018-19 through 2024-25
"""

from datetime import datetime

# Date ranges for each season
SEASONS = {
    '2018-19': {'start': datetime(2018, 8, 1), 'end': datetime(2019, 5, 31)},
    '2019-20': {'start': datetime(2019, 8, 1), 'end': datetime(2020, 7, 31)},  # Extended due to COVID
    '2020-21': {'start': datetime(2020, 8, 1), 'end': datetime(2021, 5, 31)},
    '2021-22': {'start': datetime(2021, 8, 1), 'end': datetime(2022, 5, 31)},
    '2022-23': {'start': datetime(2022, 8, 1), 'end': datetime(2023, 5, 31)},
    '2023-24': {'start': datetime(2023, 8, 1), 'end': datetime(2024, 5, 31)},
    '2024-25': {'start': datetime(2024, 8, 1), 'end': datetime(2025, 5, 31)},
}

# Premier League teams by season (20 teams each)
# Teams change due to promotion/relegation

PL_TEAMS = {
    '2018-19': {
        'Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Cardiff',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield',
        'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle',
        'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves'
    },
    '2019-20': {
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton', 'Burnley',
        'Chelsea', 'Crystal Palace', 'Everton', 'Leicester', 'Liverpool',
        'Man City', 'Man Utd', 'Newcastle', 'Norwich', 'Sheffield Utd',
        'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves'
    },
    '2020-21': {
        'Arsenal', 'Aston Villa', 'Brighton', 'Burnley', 'Chelsea',
        'Crystal Palace', 'Everton', 'Fulham', 'Leeds', 'Leicester',
        'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Sheffield Utd',
        'Southampton', 'Tottenham', 'West Brom', 'West Ham', 'Wolves'
    },
    '2021-22': {
        'Arsenal', 'Aston Villa', 'Brentford', 'Brighton', 'Burnley',
        'Chelsea', 'Crystal Palace', 'Everton', 'Leeds', 'Leicester',
        'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Norwich',
        'Southampton', 'Tottenham', 'Watford', 'West Ham', 'Wolves'
    },
    '2022-23': {
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds',
        'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle',
        'Nottm Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
    },
    '2023-24': {
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
        'Liverpool', 'Luton', 'Man City', 'Man Utd', 'Newcastle',
        'Nottm Forest', 'Sheffield Utd', 'Tottenham', 'West Ham', 'Wolves'
    },
    '2024-25': {
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
        'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Ipswich',
        'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle',
        'Nottm Forest', 'Southampton', 'Tottenham', 'West Ham', 'Wolves'
    },
}

# Teams to explicitly exclude (lower leagues, women's teams, etc.)
EXCLUDED_TEAMS = {
    'Leeds United U21', 'Arsenal U21', 'Chelsea U21', 'Manchester City U21',
    'Arsenal Women', 'Chelsea Women', 'Man City Women',
    'Sunderland', 'Derby', 'Birmingham', 'Reading', 'QPR', 'Middlesbrough',
    'Stoke', 'Blackburn', 'Swansea', 'Wigan', 'Bolton', 'Hull',
}

# Output directory structure
OUTPUT_DIR = 'historical_output'
LOG_FILE = 'download_log.txt'
SUMMARY_FILE = 'matches_summary.csv'

# Betfair API parameters
MARKET_TYPE = 'MATCH_ODDS'
COUNTRY_CODE = 'GB'
FILE_TYPE = 'M'  # Market data

# Valid Betfair tick sizes for rounding odds
BETFAIR_TICKS = [
    1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10,
    1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20,
    1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.30,
    1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.40,
    1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.50,
    1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.60,
    1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.70,
    1.71, 1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.80,
    1.81, 1.82, 1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.90,
    1.91, 1.92, 1.93, 1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.00,
] + [i/100 for i in range(202, 301, 2)] + [i/100 for i in range(305, 401, 5)] + \
    [i/10 for i in range(42, 51)] + [i for i in range(6, 11)] + \
    [i/2 for i in range(22, 41)] + [i for i in range(22, 31)] + \
    [i/2 for i in range(62, 101, 5)] + list(range(60, 101, 10)) + list(range(110, 1001, 10))


# ============================================================================
# AUTHENTICATION
# ============================================================================

def get_trading_client(timeout=60):
    """
    Get authenticated Betfair API client using external credentials

    Usage:
        trading = get_trading_client()
        # ... use client ...

    Args:
        timeout: Request timeout in seconds (default: 60)

    Note:
        Imports from external login.py (not in this repo)
        Returns betfairlightweight.APIClient trading client from Flumine framework
    """
    import sys, os
    login_dir = os.environ.get('BETFAIR_LOGIN_DIR')  # dir containing login.py (credentials, not in repo)
    if not login_dir or not os.path.isdir(login_dir):
        raise RuntimeError("Set BETFAIR_LOGIN_DIR to the directory containing login.py")
    sys.path.insert(0, login_dir)
    from login import framework

    fw = framework()

    # Try to get the betfairlightweight client from Flumine
    # The framework wraps clients, so we need to find the underlying betfairlightweight.APIClient

    try:
        # Get the default Betfair client from Flumine framework (get_default in flumine >=2.x)
        clients = fw.clients
        betfair_client = clients.get_default() if hasattr(clients, 'get_default') else clients.get_betfair_default()

        # Try common attribute names for the wrapped client
        for attr in ['betting_client', 'client', '_client', 'api', 'trading', '_trading']:
            if hasattr(betfair_client, attr):
                client = getattr(betfair_client, attr)
                # Verify it's a betfairlightweight client
                if hasattr(client, 'historic'):
                    # login.py hardcodes a Windows certs path; repoint at certs/ beside login.py
                    certs = os.path.join(login_dir, 'certs')
                    if not os.path.isdir(certs):
                        raise RuntimeError(f"certs dir not found: {certs}")
                    client.certs = certs
                    # Set timeout for requests
                    if hasattr(client, 'request_timeout'):
                        client.request_timeout = timeout
                    # Also try to set on the session if available
                    if hasattr(client, 'session') and hasattr(client.session, 'timeout'):
                        client.session.timeout = timeout
                    return client

        # If we can't find it, show what's available for debugging
        attrs = [a for a in dir(betfair_client) if not a.startswith('__')]
        raise AttributeError(
            f"Could not find betfairlightweight client in BetfairClient.\n"
            f"Available attributes: {attrs}\n"
            f"You may need to use get_trading_client_direct() instead."
        )

    except Exception as e:
        print(f"Error accessing framework client: {e}")
        print("\nTrying alternative: creating betfairlightweight client directly...")
        raise


def get_trading_client_direct(username, password, app_key, certs_path):
    """
    Create betfairlightweight.APIClient directly without Flumine wrapper

    Usage:
        trading = get_trading_client_direct(
            username='your_username',
            password='your_password',
            app_key='your_app_key',
            certs_path=r'path/to/certs'
        )

    Args:
        username: Betfair account username
        password: Betfair account password
        app_key: Betfair API application key
        certs_path: Path to SSL certificate directory

    Returns:
        betfairlightweight.APIClient: Authenticated trading client
    """
    import betfairlightweight

    trading = betfairlightweight.APIClient(
        username=username,
        password=password,
        app_key=app_key,
        certs=certs_path
    )

    trading.login()
    return trading

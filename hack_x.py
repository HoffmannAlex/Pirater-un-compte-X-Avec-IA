"""
X (Twitter) Account Security Tester
Educational use only - Version 2025.1
"""

import os
import sys
import json
import time
import random
import asyncio
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, List

import tweepy
from dotenv import load_dotenv
from proxy_manager import ProxyManager
from tor_manager import TorManager
from monitoring import OperationMonitor
from request_manager import RequestManager
from logging_config import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger('x_security_tester')

class XSecurityTester:
    """X (Twitter) Account Security Tester"""
    
    def __init__(self, username: str):
        """Initialize X Security Tester"""
        self.username = username.lstrip('@')
        self.client = self._initialize_twitter_client()
        self.monitor = OperationMonitor()
        
    def _initialize_twitter_client(self):
        """Initialize Twitter API client"""
        try:
            client = tweepy.Client(
                consumer_key=os.getenv('CONSUMER_KEY'),
                consumer_secret=os.getenv('CONSUMER_SECRET'),
                access_token=os.getenv('ACCESS_TOKEN'),
                access_token_secret=os.getenv('ACCESS_TOKEN_SECRET'),
                wait_on_rate_limit=True
            )
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            sys.exit(1)
    
    async def test_credentials(self, username: str, password: str) -> bool:
        """Test X (Twitter) credentials"""
        try:
            # This is a placeholder for actual credential testing logic
            # In a real implementation, you would need to handle X's authentication flow
            logger.debug(f"Testing credentials for {username}")
            await asyncio.sleep(0.1)  # Simulate network delay
            return False  # Default to False for safety
        except Exception as e:
            logger.error(f"Error testing credentials: {e}")
            return False

def display_banner():
    """Display tool banner with security disclaimer"""
    print("\033[1;32m")
    print("X (Twitter) Account Security Tester")
    print("Version 2025.1 - Educational Use Only")
    print("=" * 50)
    print("FOR AUTHORIZED SECURITY TESTING ONLY")
    print("UNAUTHORIZED USE IS STRICTLY PROHIBITED")
    print("\033[1;37m")

def validate_file(file_path: str, description: str) -> None:
    """Validate file existence"""
    if file_path and not os.path.isfile(file_path):
        logger.error(f"{description} not found: {file_path}")
        sys.exit(1)
    elif file_path:
        logger.info(f"{description} found: {file_path}")

def show_usage():
    """Display tool usage instructions"""
    print("\033[1;34m")  
    print("""
Usage Examples:
---------------
1. Basic Security Test:
   python hack_x.py --username @target_user --password-list passwords.txt

2. Anonymous Testing with Tor:
   python hack_x.py --username @target_user --password-list passwords.txt --use-tor

3. Proxy-Based Testing:
   python hack_x.py --username @target_user --password-list passwords.txt --proxy-list proxies.txt

4. Multi-threaded Testing:
   python hack_x.py --username @target_user --password-list passwords.txt --threads 4
    """)
    print("\033[1;37m")  

def setup_proxy(args) -> Optional[ProxyManager]:
    """Configure proxy manager if requested"""
    if args.proxy_list:
        proxy_manager = ProxyManager(proxy_file=args.proxy_list)
        if not proxy_manager.validate_all_proxies():
            logger.warning("Some proxies failed validation")
        logger.info("Proxies loaded successfully")
        return proxy_manager
    return None

def setup_tor(args) -> None:
    """Configure Tor for anonymous testing"""
    if args.use_tor:
        logger.info("Configuring Tor for anonymous testing...")
        tor_manager = TorManager()
        if not tor_manager.verify_tor_connection():
            logger.error("Tor connection verification failed")
            sys.exit(1)
        logger.info("Tor connection established")

def validate_arguments(args) -> bool:
    """Validate command line arguments"""
    errors = []
    
    if not args.username:
        errors.append("X username is required (e.g., @username)")
    
    if not args.password_list:
        errors.append("Password list is required for testing")
    elif not os.path.exists(args.password_list):
        errors.append(f"Password list file not found: {args.password_list}")
    
    if args.proxy_list and not os.path.exists(args.proxy_list):
        errors.append(f"Proxy list file not found: {args.proxy_list}")
    
    if args.threads < 1 or args.threads > 10:
        errors.append("Thread count must be between 1 and 10")
    
    # Validate API credentials
    required_env_vars = [
        'CONSUMER_KEY', 'CONSUMER_SECRET',
        'ACCESS_TOKEN', 'ACCESS_TOKEN_SECRET'
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        errors.append(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    if errors:
        for error in errors:
            logger.error(error)
        return False
    return True

def handle_arguments():
    """Parse and handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="X (Twitter) Account Security Tester - Educational Use Only"
    )
    parser.add_argument("--username", required=True, help="Target X (Twitter) username for security testing")
    parser.add_argument("--password-list", required=True, help="Path to password list file for testing")
    parser.add_argument("--timeout", type=int, default=3600, help="Testing timeout in seconds")
    parser.add_argument("--use-tor", action="store_true", help="Enable Tor for anonymous testing")
    parser.add_argument("--proxy-list", help="Path to proxy list file")
    parser.add_argument("--threads", type=int, default=1, help="Number of testing threads")
    parser.add_argument("--min-delay", type=float, default=1.0, help="Minimum delay between requests")
    parser.add_argument("--max-delay", type=float, default=3.0, help="Maximum delay between requests")
    parser.add_argument("--output", default="security_results.txt", help="Output file for test results")
    
    args = parser.parse_args()
    if not validate_arguments(args):
        exit(1)
    return args

async def main():
    """Main function for security testing tool"""
    display_banner()
    args = handle_arguments()
    
    try:
        # Security disclaimer
        print("\n🔒 SECURITY DISCLAIMER:")
        print("🔒 This tool is for educational and authorized security testing only")
        print("🔒 Unauthorized use may violate laws and terms of service")
        print("🔒 You are responsible for ensuring proper authorization")
        
        confirm = input("\nType 'AUTHORIZED' to continue: ")
        if confirm != "AUTHORIZED":
            print("Operation cancelled - Authorization required")
            return
        
        monitor = OperationMonitor()
        request_manager = RequestManager(min_delay=args.min_delay, max_delay=args.max_delay)
        security_tester = AISecurityTester()
        
        # Load passwords for testing
        with open(args.password_list, 'r') as f:
            passwords = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(passwords)} passwords for security testing")
        
        if args.proxy_list:
            proxy_manager = ProxyManager(proxy_file=args.proxy_list)
            await proxy_manager.initialize_proxies()
            request_manager.set_proxy_manager(proxy_manager)
            logger.info("Proxy configuration enabled")
        
        if args.use_tor:
            tor_manager = TorManager()
            if not tor_manager.ensure_tor_running():
                logger.error("Failed to initialize Tor. Exiting...")
                return
            logger.info("Tor anonymity enabled")
        
        monitor.start_monitoring()
        
        # Conduct security testing
        logger.info(f"Starting security assessment for user: {args.username}")
        result = await security_tester.conduct_ai_security_test(
            username=args.username,
            max_attempts=len(passwords),
            delay=args.min_delay
        )
        
        monitor.stop_monitoring()
        
        # Save results
        with open(args.output, 'w') as f:
            if result.success:
                f.write(f"SECURITY ALERT: Weak password detected\n")
                f.write(f"Username: {args.username}\n")
                f.write(f"Password: {result.password}\n")
                f.write(f"Attempts: {result.attempts}\n")
                f.write(f"Duration: {result.duration:.2f}s\n")
            else:
                f.write(f"SECURITY ASSESSMENT COMPLETE\n")
                f.write(f"Username: {args.username}\n")
                f.write(f"Result: No weak passwords detected\n")
                f.write(f"Tests conducted: {result.attempts}\n")
                f.write(f"Duration: {result.duration:.2f}s\n")
        
        if result.success:
            logger.info(f"Security issue identified! Results saved to {args.output}")
            logger.info(f"Weak password: {result.password}")
        else:
            logger.info("Security assessment completed - no issues detected")
            logger.info(f"Results saved to {args.output}")
            
    except KeyboardInterrupt:
        logger.info("\nSecurity testing interrupted by user")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"Unexpected error during security testing: {str(e)}")
        monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
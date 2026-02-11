# -*- coding: utf-8 -*-
"""
================================================================================
EXTERNAL FEATURE EXTRACTION MODULE
================================================================================
Extracts features that require external API calls, web scraping, and WHOIS lookups.

Features extracted:
- UrlIsLive: Check if URL is accessible
- DomainAge: Get domain age via WHOIS
- NoOfImgaeNonOrigional: Count non-original images on webpage
- HasTitle: Check if page has HTML title
- Additional webpage features as needed

Author: Yousef
Date: 2026-02-10
================================================================================
"""

import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import Dict, Any, Optional
import socket
import ssl
import whois
from datetime import datetime
import time
import warnings

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Request configuration
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 2
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def check_url_is_live(url: str) -> int:
    """
    Check if a URL is accessible and live.
    
    Args:
        url: The URL to check
        
    Returns:
        1 if live, 0 if not accessible, -1 if error
    """
    try:
        # Try to make a HEAD request first (faster)
        response = requests.head(
            url, 
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            verify=False  # Ignore SSL verification for testing
        )
        
        # If HEAD fails, try GET
        if response.status_code >= 400:
            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=False
            )
        
        # Consider 2xx and 3xx as live
        if 200 <= response.status_code < 400:
            return 1
        else:
            return 0
            
    except requests.exceptions.SSLError:
        # SSL errors often indicate phishing sites with bad certificates
        logger.debug(f"SSL Error for {url} - marking as live but suspicious")
        return 0
    except (requests.exceptions.RequestException, socket.error) as e:
        logger.debug(f"URL not accessible: {url} - {str(e)}")
        return 0
    except Exception as e:
        logger.warning(f"Error checking URL live status: {e}")
        return -1


def get_domain_age(url: str) -> float:
    """
    Get the age of the domain in days via WHOIS lookup.
    
    Args:
        url: The URL to check
        
    Returns:
        Age in days, or -1 if unable to determine
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Perform WHOIS lookup
        w = whois.whois(domain)
        
        # Get creation date
        creation_date = w.creation_date
        
        # Handle list of dates (some WHOIS return multiple dates)
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        
        if creation_date:
            age_days = (datetime.now() - creation_date).days
            return float(age_days)
        else:
            logger.debug(f"No creation date found for {domain}")
            return -1
            
    except Exception as e:
        logger.debug(f"WHOIS lookup failed for {url}: {e}")
        return -1


def fetch_webpage_content(url: str) -> Optional[BeautifulSoup]:
    """
    Fetch and parse webpage HTML content.
    
    Args:
        url: The URL to fetch
        
    Returns:
        BeautifulSoup object or None if failed
    """
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            verify=False
        )
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
        else:
            logger.debug(f"Failed to fetch {url}: Status {response.status_code}")
            return None
            
    except Exception as e:
        logger.debug(f"Error fetching webpage {url}: {e}")
        return None


def check_has_title(soup: Optional[BeautifulSoup]) -> int:
    """
    Check if webpage has a title tag.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        1 if has title, 0 otherwise
    """
    if soup is None:
        return 0
    
    try:
        title = soup.find('title')
        if title and title.string and len(title.string.strip()) > 0:
            return 1
        else:
            return 0
    except Exception as e:
        logger.debug(f"Error checking title: {e}")
        return 0


def count_non_original_images(url: str, soup: Optional[BeautifulSoup]) -> int:
    """
    Count images from external domains (non-original).
    
    Args:
        url: Base URL for determining domain
        soup: BeautifulSoup object
        
    Returns:
        Count of external images, or 0 if unable to determine
    """
    if soup is None:
        return 0
    
    try:
        parsed_base = urlparse(url)
        base_domain = parsed_base.netloc
        
        # Find all image tags
        images = soup.find_all('img')
        external_count = 0
        
        for img in images:
            src = img.get('src', '')
            
            if not src:
                continue
            
            # Convert relative URLs to absolute
            if src.startswith('//'):
                src = 'http:' + src
            elif src.startswith('/') or not src.startswith('http'):
                src = urljoin(url, src)
            
            # Parse image URL
            try:
                img_parsed = urlparse(src)
                img_domain = img_parsed.netloc
                
                # Check if domain is different
                if img_domain and img_domain != base_domain:
                    external_count += 1
            except:
                continue
        
        return external_count
        
    except Exception as e:
        logger.debug(f"Error counting external images: {e}")
        return 0


def get_ssl_details(url: str) -> Dict[str, Any]:
    """
    Get SSL certificate details (Issuer, Age).
    """
    ssl_info = {'issuer': 'Unknown', 'age_days': -1, 'valid': 0}
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc
        if ':' in hostname: hostname = hostname.split(':')[0]
        
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                
                # Issuer
                issuer = dict(x[0] for x in cert['issuer'])
                org = issuer.get('organizationName') or issuer.get('commonName') or 'Unknown'
                ssl_info['issuer'] = org
                
                # Age
                notBefore = datetime.strptime(cert['notBefore'], r'%b %d %H:%M:%S %Y %Z')
                age = (datetime.now() - notBefore).days
                ssl_info['age_days'] = age
                ssl_info['valid'] = 1
                
    except Exception as e:
        logger.debug(f"SSL Check failed for {url}: {e}")
        
    return ssl_info


def get_ip_reputation(url: str) -> Dict[str, Any]:
    """
    Get IP and ASN reputation using ip-api.com (free).
    """
    ip_info = {'ip': 'Unknown', 'msg': 'Failed', 'country': 'Unknown', 'isp': 'Unknown'}
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc
        if ':' in hostname: hostname = hostname.split(':')[0]
        
        # 1. Resolve IP
        ip_addr = socket.gethostbyname(hostname)
        ip_info['ip'] = ip_addr
        
        # 2. Get ASN/ISP info
        response = requests.get(f'http://ip-api.com/json/{ip_addr}?fields=status,message,country,isp,org,as', timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                ip_info['msg'] = 'Success'
                ip_info['country'] = data.get('country', 'Unknown')
                ip_info['isp'] = data.get('isp', 'Unknown')
                ip_info['org'] = data.get('org', 'Unknown')
                ip_info['asn'] = data.get('as', 'Unknown')

    except Exception as e:
        logger.debug(f"IP Reputation check failed: {e}")
        
    return ip_info


def extract_external_features(url: str, fetch_content: bool = True) -> Dict[str, Any]:
    """
    Extract all external features for a URL.
    
    Args:
        url: The URL to analyze
        fetch_content: If True, fetch webpage content (slower but more features)
        
    Returns:
        Dictionary of external features
    """
    features = {}
    
    logger.debug(f"Extracting external features for: {url}")
    
    # 1. Check if URL is live
    features['UrlIsLive'] = check_url_is_live(url)
    
    # 2. Get domain age
    features['DomainAge'] = get_domain_age(url)
    
    # 3. Fetch webpage content if URL is live and fetch_content is True
    soup = None
    if fetch_content and features['UrlIsLive'] == 1:
        soup = fetch_webpage_content(url)
    
    # 4. Check if page has title
    features['HasTitle'] = check_has_title(soup)
    
    # 5. Count non-original images
    features['NoOfImgaeNonOrigional'] = count_non_original_images(url, soup)
    
    # Additional derived features
    features['HasValidSSL'] = 1 if url.startswith('https://') else 0
    
    # 6. Advanced SSL (New)
    ssl_data = get_ssl_details(url)
    features['SSLIssuer'] = ssl_data['issuer']
    features['SSLAgeDays'] = ssl_data['age_days']
    
    # 7. IP Reputation (New)
    ip_data = get_ip_reputation(url)
    features['HostingIP'] = ip_data['ip']
    features['HostingISP'] = ip_data['isp']
    features['HostingCountry'] = ip_data['country']
    
    # Domain age categories
    if features['DomainAge'] != -1:
        if features['DomainAge'] < 30:
            features['DomainAgeDays'] = features['DomainAge']
            features['IsSuspiciouslyNew'] = 1  # Very new domains are suspicious
        elif features['DomainAge'] < 365:
            features['DomainAgeDays'] = features['DomainAge']
            features['IsSuspiciouslyNew'] = 0
        else:
            features['DomainAgeDays'] = features['DomainAge']
            features['IsSuspiciouslyNew'] = 0
    else:
        features['DomainAgeDays'] = -1
        features['IsSuspiciouslyNew'] = 0
    
    logger.debug(f"External features extracted: {features}")
    
    return features


def extract_external_features_batch(urls: list, fetch_content: bool = True, delay: float = 0.5) -> list:
    """
    Extract external features for multiple URLs with rate limiting.
    
    Args:
        urls: List of URLs to process
        fetch_content: If True, fetch webpage content
        delay: Delay between requests to avoid rate limiting
        
    Returns:
        List of feature dictionaries
    """
    results = []
    
    for i, url in enumerate(urls):
        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        
        features = extract_external_features(url, fetch_content=fetch_content)
        results.append(features)
        
        # Add delay to avoid overwhelming servers
        if i < len(urls) - 1:
            time.sleep(delay)
    
    return results


if __name__ == "__main__":
    # Test the module
    test_urls = [
        "https://www.google.com",
        "https://www.github.com",
        "http://192.168.1.1"
    ]
    
    print("Testing External Feature Extraction")
    print("=" * 70)
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        features = extract_external_features(url, fetch_content=True)
        for key, value in features.items():
            print(f"  {key}: {value}")

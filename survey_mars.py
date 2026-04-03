import os
import requests
import uuid
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class SurveyMarsClient:
    def __init__(self, account_id=None, secret=None):
        self.base_url = "https://api.surveymars.com/v1"
        self.account_id = account_id or os.environ.get("SURVEYMARS_ACCOUNT_ID")
        self.secret = secret or os.environ.get("SURVEYMARS_SECRET")
        self.access_token = None
        self.refresh_token = None

        if not self.account_id or not self.secret:
            raise ValueError(
                "SurveyMars credentials not found. "
                "Provide account_id/secret or set SURVEYMARS_ACCOUNT_ID/SURVEYMARS_SECRET in .env or environment."
            )

    def generate_request_id(self):
        """Generates a unique X-Request-Id (UUID max 36 chars)"""
        return str(uuid.uuid4())

    def authenticate(self):
        """Generates a brand new access and refresh token using Account ID and Secret."""
        print("Authenticating with Account ID and Secret...")
        url = f"{self.base_url}/authenticate"
        
        headers = {
            "X-Request-Id": self.generate_request_id(),
            "Content-Type": "application/json"
        }
        
        # Matches the exact body required by your docs
        payload = {
            "id": self.account_id,
            "credential": self.secret
        }
        
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        if response.status_code == 200 and data.get("success"):
            self.access_token = data["data"]["access_token"]["token"]
            self.refresh_token = data["data"]["refresh_token"]["token"]
            print("Successfully authenticated! New tokens acquired.")
        else:
            raise Exception(f"Authentication Failed! Error: {data}")

    def make_request(self, method, endpoint, params=None, json_data=None):
        """A helper method to make authenticated requests to the API."""
        
        # 1. If we don't have an access token yet, log in first
        if not self.access_token:
            self.authenticate()

        url = f"{self.base_url}/{endpoint}"
        
        # 2. Build the headers with the access token
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-Request-Id": self.generate_request_id(),
            "Content-Type": "application/json"
        }
        
        # 3. Send the request
        response = requests.request(method, url, headers=headers, params=params, json=json_data)
        
        # 4. Handle Rate Limiting (HTTP 429)
        if response.status_code == 429:
            print("Rate limit exceeded (HTTP 429). Waiting 1 second before retrying...")
            time.sleep(1)
            return self.make_request(method, endpoint, params, json_data)
            
        data = response.json()
        
        # 5. Handle Expired/Invalid Tokens Automatically
        if not data.get("success") and "error" in data:
            error_code = data["error"]["id"]
            
            # 2004 = Expired, 2005 = Invalid
            if error_code in [2004, 2005]:
                print(f"Token error ({error_code}). Generating fresh tokens...")
                self.authenticate() # Just log in again from scratch
                
                # Retry the exact same request with the brand new token
                return self.make_request(method, endpoint, params, json_data)
            else:
                print(f"API Error {error_code}: {data['error']['message']}")
                
        return data
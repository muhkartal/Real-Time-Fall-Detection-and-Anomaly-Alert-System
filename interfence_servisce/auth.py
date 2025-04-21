#!/usr/bin/env python3
"""
Authentication and Authorization middleware for EdgeVision-Guard.

This module provides JWT-based authentication and role-based access control
for the EdgeVision-Guard API service.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable

import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


class User(BaseModel):
    """User model for authentication."""
    
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    roles: List[str] = []


class Token(BaseModel):
    """Token model for authentication."""
    
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    """Token data model for decoded JWT."""
    
    username: str
    roles: List[str] = []
    exp: Optional[int] = None


class JWTBearer(HTTPBearer):
    """Bearer token authentication for JWT."""
    
    def __init__(self, auto_error: bool = True):
        """
        Initialize the JWT bearer authentication.
        
        Args:
            auto_error: Whether to auto-error on authentication failure
        """
        super(JWTBearer, self).__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[TokenData]:
        """
        Verify and decode JWT token.
        
        Args:
            request: FastAPI request
        
        Returns:
            Decoded token data
        
        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication if disabled
        if not AUTH_ENABLED:
            # Return a default token with admin role for development
            return TokenData(
                username="dev",
                roles=["admin", "viewer"],
                exp=int(time.time()) + 3600,
            )
        
        # Get credentials
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        # Check if we have credentials
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme",
                )
            
            # Verify token
            token_data = self.verify_jwt(credentials.credentials)
            if not token_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                )
            
            return token_data
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization code",
            )
    
    def verify_jwt(self, jwt_token: str) -> Optional[TokenData]:
        """
        Verify JWT token.
        
        Args:
            jwt_token: JWT token to verify
        
        Returns:
            Decoded token data if valid, None otherwise
        """
        try:
            # Decode token
            payload = jwt.decode(jwt_token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Check expiration
            if payload.get("exp") < time.time():
                return None
            
            # Return token data
            return TokenData(
                username=payload.get("sub", ""),
                roles=payload.get("roles", []),
                exp=payload.get("exp"),
            )
        except jwt.PyJWTError:
            return None


# Create security dependency
security = JWTBearer()


# Role-based access control
def require_roles(required_roles: List[str]) -> Callable:
    """
    Require specific roles for access.
    
    Args:
        required_roles: List of required role names
    
    Returns:
        Dependency function for role-based access control
    """
    async def role_checker(token_data: TokenData = Depends(security)) -> TokenData:
        """
        Check if user has required roles.
        
        Args:
            token_data: Decoded token data
        
        Returns:
            Token data if user has required roles
        
        Raises:
            HTTPException: If user doesn't have required roles
        """
        # Skip role check if authentication is disabled
        if not AUTH_ENABLED:
            return token_data
        
        # Check if user has admin role (which has all permissions)
        if "admin" in token_data.roles:
            return token_data
        
        # Check if user has any of the required roles
        for role in required_roles:
            if role in token_data.roles:
                return token_data
        
        # If we get here, the user doesn't have any of the required roles
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    
    return role_checker


# Function to create access token
def create_access_token(
    username: str, 
    roles: List[str] = [], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        username: Username to encode in token
        roles: User roles to encode in token
        expires_delta: Token expiration time
    
    Returns:
        JWT token string
    """
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    
    # Create payload
    payload = {
        "sub": username,
        "roles": roles,
        "exp": expire,
        "iat": datetime.utcnow(),
    }
    
    # Encode token
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# Function to get current user
async def get_current_user(token_data: TokenData = Depends(security)) -> User:
    """
    Get current authenticated user.
    
    Args:
        token_data: Decoded token data
    
    Returns:
        User object
    """
    return User(
        username=token_data.username,
        roles=token_data.roles,
    )


# Function to set up authentication for FastAPI app
def setup_auth(app: FastAPI) -> None:
    """
    Set up authentication for FastAPI app.
    
    Args:
        app: FastAPI application
    """
    if not AUTH_ENABLED:
        logger.warning("Authentication is DISABLED. Set AUTH_ENABLED=true to enable.")
        return
    
    # Add login endpoint
    @app.post("/auth/token", response_model=Token)
    async def login(username: str, password: str) -> Token:
        """
        Login endpoint to get access token.
        
        Args:
            username: Username
            password: Password
        
        Returns:
            Token object
        
        Raises:
            HTTPException: If authentication fails
        """
        # In a real application, you would validate credentials against a database
        # For simplicity, we'll use hardcoded values here
        if username == "admin" and password == "password":
            # Create access token
            expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
            access_token = create_access_token(
                username=username,
                roles=["admin"],
                expires_delta=expires,
            )
            
            # Return token
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_at=datetime.utcnow() + expires,
            )
        elif username == "viewer" and password == "password":
            # Create access token with viewer role
            expires = timedelta(minutes=JWT_EXPIRATION_MINUTES)
            access_token = create_access_token(
                username=username,
                roles=["viewer"],
                expires_delta=expires,
            )
            
            # Return token
            return Token(
                access_token=access_token,
                token_type="bearer",
                expires_at=datetime.utcnow() + expires,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Add user info endpoint
    @app.get("/auth/me", response_model=User)
    async def get_user_info(current_user: User = Depends(get_current_user)) -> User:
        """
        Get current user info.
        
        Args:
            current_user: Current authenticated user
        
        Returns:
            User object
        """
        return current_user
    
    logger.info("Authentication middleware initialized")


if __name__ == "__main__":
    # Generate a random JWT secret key
    import secrets
    print(f"Generated JWT secret key: {secrets.token_hex(32)}")
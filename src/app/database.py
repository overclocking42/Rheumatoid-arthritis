"""
Database module for user authentication and prediction history management.
Uses SQLite for persistent storage.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import hashlib
import secrets


class Database:
    """SQLite database manager for users and prediction history."""
    
    def __init__(self, db_path: str = "data/app_users.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        """)
        
        # Prediction history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_type TEXT NOT NULL,  -- 'lab', 'xray', 'combined'
            input_data TEXT NOT NULL,  -- JSON string
            result_data TEXT NOT NULL,  -- JSON string
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Report history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS report_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            report_title TEXT NOT NULL,
            report_content TEXT NOT NULL,  -- HTML/Markdown content
            lab_data TEXT,  -- JSON of lab inputs
            imaging_data TEXT,  -- JSON of imaging inputs
            prediction_data TEXT,  -- JSON of predictions
            report_type TEXT,  -- 'manual', 'bulk_csv', 'bulk_image'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_history_user_date ON prediction_history(user_id, created_at)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_report_user_date ON report_history(user_id, created_at)
        """)
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt using PBKDF2."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        
        return password_hash, salt
    
    def create_user(self, email: str, password: str, full_name: str = "") -> Tuple[bool, str]:
        """
        Create a new user account.
        
        Args:
            email: User email address
            password: User password (will be hashed)
            full_name: User's full name
        
        Returns:
            (success: bool, message: str)
        """
        # Validate email
        if not email or '@' not in email:
            return False, "Invalid email address"
        
        # Validate password
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters"
        
        # Validate name
        if not full_name or len(full_name.strip()) < 2:
            return False, "Name must be at least 2 characters"
        
        try:
            password_hash, salt = self._hash_password(password)
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO users (email, full_name, password_hash, salt)
            VALUES (?, ?, ?, ?)
            """, (email.lower(), full_name.strip(), password_hash, salt))
            
            conn.commit()
            conn.close()
            return True, "Account created successfully"
        
        except sqlite3.IntegrityError:
            return False, "Email already registered"
        except Exception as e:
            return False, f"Error creating account: {str(e)}"
    
    def authenticate_user(self, email: str, password: str) -> Tuple[bool, Optional[int], Optional[str], str]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User email address
            password: User password
        
        Returns:
            (success: bool, user_id: Optional[int], full_name: Optional[str], message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT id, password_hash, salt, full_name FROM users WHERE email = ?
            """, (email.lower(),))
            
            result = cursor.fetchone()
            
            if result is None:
                return False, None, None, "Email not found"
            
            user_id, stored_hash, salt, full_name = result['id'], result['password_hash'], result['salt'], result['full_name']
            
            # Verify password
            password_hash, _ = self._hash_password(password, salt)
            
            if password_hash != stored_hash:
                return False, None, None, "Incorrect password"
            
            # Update last login
            cursor.execute("""
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
            """, (user_id,))
            conn.commit()
            conn.close()
            
            return True, user_id, full_name, "Login successful"
        
        except Exception as e:
            return False, None, None, f"Error logging in: {str(e)}"
    
    def user_exists(self, email: str) -> bool:
        """Check if user exists by email."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM users WHERE email = ?", (email.lower(),))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception:
            return False
    
    def save_prediction(
        self,
        user_id: int,
        prediction_type: str,
        input_data: Dict,
        result_data: Dict,
        confidence: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Save prediction to history.
        
        Args:
            user_id: User ID
            prediction_type: Type of prediction ('lab', 'xray', 'combined')
            input_data: Dictionary of input parameters
            result_data: Dictionary of prediction results
            confidence: Confidence score (0-1 or 0-100)
        
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO prediction_history 
            (user_id, prediction_type, input_data, result_data, confidence)
            VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                prediction_type,
                json.dumps(input_data),
                json.dumps(result_data),
                confidence
            ))
            
            conn.commit()
            conn.close()
            return True, "Prediction saved to history"
        
        except Exception as e:
            return False, f"Error saving prediction: {str(e)}"
    
    def get_user_history(
        self,
        user_id: int,
        prediction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get prediction history for a user.
        
        Args:
            user_id: User ID
            prediction_type: Filter by type ('lab', 'xray', 'combined') or None for all
            limit: Maximum number of records to return
        
        Returns:
            List of prediction history records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if prediction_type:
                cursor.execute("""
                SELECT * FROM prediction_history 
                WHERE user_id = ? AND prediction_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """, (user_id, prediction_type, limit))
            else:
                cursor.execute("""
                SELECT * FROM prediction_history 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """, (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to list of dictionaries with parsed JSON
            history = []
            for row in rows:
                history.append({
                    'id': row['id'],
                    'prediction_type': row['prediction_type'],
                    'input_data': json.loads(row['input_data']),
                    'result_data': json.loads(row['result_data']),
                    'confidence': row['confidence'],
                    'created_at': row['created_at']
                })
            
            return history
        
        except Exception as e:
            print(f"Error retrieving history: {str(e)}")
            return []
    
    def delete_prediction(self, prediction_id: int, user_id: int) -> Tuple[bool, str]:
        """
        Delete a prediction record (ownership verification included).
        
        Args:
            prediction_id: Prediction ID
            user_id: User ID (for verification)
        
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Verify ownership
            cursor.execute("""
            SELECT user_id FROM prediction_history WHERE id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if result is None or result['user_id'] != user_id:
                return False, "Permission denied"
            
            # Delete record
            cursor.execute("""
            DELETE FROM prediction_history WHERE id = ?
            """, (prediction_id,))
            
            conn.commit()
            conn.close()
            return True, "Prediction deleted"
        
        except Exception as e:
            return False, f"Error deleting prediction: {str(e)}"
    
    def clear_user_history(self, user_id: int) -> Tuple[bool, str]:
        """
        Clear all prediction history for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True, "History cleared"
        except Exception as e:
            return False, f"Error clearing history: {str(e)}"
    
    def get_user_stats(self, user_id: int) -> Dict:
        """
        Get statistics for a user's predictions.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary with stats
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get counts by type
            cursor.execute("""
            SELECT prediction_type, COUNT(*) as count 
            FROM prediction_history 
            WHERE user_id = ? 
            GROUP BY prediction_type
            """, (user_id,))
            
            type_counts = {row['prediction_type']: row['count'] for row in cursor.fetchall()}
            
            # Get total
            cursor.execute("""
            SELECT COUNT(*) as total FROM prediction_history WHERE user_id = ?
            """, (user_id,))
            
            total = cursor.fetchone()['total']
            
            # Get first and last prediction dates
            cursor.execute("""
            SELECT MIN(created_at) as first, MAX(created_at) as last 
            FROM prediction_history 
            WHERE user_id = ?
            """, (user_id,))
            
            dates = cursor.fetchone()
            conn.close()
            
            return {
                'total_predictions': total,
                'by_type': type_counts,
                'first_prediction': dates['first'],
                'last_prediction': dates['last']
            }
        
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {'total_predictions': 0, 'by_type': {}, 'first_prediction': None, 'last_prediction': None}
    
    def save_report(
        self,
        user_id: int,
        report_title: str,
        report_content: str,
        lab_data: Optional[Dict] = None,
        imaging_data: Optional[Dict] = None,
        prediction_data: Optional[Dict] = None,
        report_type: str = 'manual'
    ) -> Tuple[bool, str]:
        """
        Save a clinical report to history.
        
        Args:
            user_id: User ID
            report_title: Report title
            report_content: Report HTML/Markdown content
            lab_data: Lab input data (JSON)
            imaging_data: Imaging input data (JSON)
            prediction_data: Prediction results (JSON)
            report_type: 'manual', 'bulk_csv', or 'bulk_image'
        
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO report_history 
            (user_id, report_title, report_content, lab_data, imaging_data, prediction_data, report_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                report_title,
                report_content,
                json.dumps(lab_data) if lab_data else None,
                json.dumps(imaging_data) if imaging_data else None,
                json.dumps(prediction_data) if prediction_data else None,
                report_type
            ))
            
            conn.commit()
            conn.close()
            return True, "Report saved successfully"
        
        except Exception as e:
            return False, f"Error saving report: {str(e)}"
    
    def get_user_reports(self, user_id: int, limit: int = 100) -> List[Dict]:
        """
        Get all reports for a user, sorted by date (newest first).
        
        Args:
            user_id: User ID
            limit: Maximum reports to retrieve
        
        Returns:
            List of report dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT * FROM report_history 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            reports = []
            for row in rows:
                reports.append({
                    'id': row['id'],
                    'report_title': row['report_title'],
                    'report_type': row['report_type'],
                    'created_at': row['created_at'],
                    'report_content': row['report_content'],
                    'lab_data': json.loads(row['lab_data']) if row['lab_data'] else None,
                    'imaging_data': json.loads(row['imaging_data']) if row['imaging_data'] else None,
                    'prediction_data': json.loads(row['prediction_data']) if row['prediction_data'] else None
                })
            
            return reports
        
        except Exception as e:
            print(f"Error retrieving reports: {str(e)}")
            return []
    
    def delete_report(self, report_id: int, user_id: int) -> Tuple[bool, str]:
        """
        Delete a report (with ownership verification).
        
        Args:
            report_id: Report ID
            user_id: User ID (for verification)
        
        Returns:
            (success: bool, message: str)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Verify ownership
            cursor.execute("""
            SELECT user_id FROM report_history WHERE id = ?
            """, (report_id,))
            
            result = cursor.fetchone()
            if result is None or result['user_id'] != user_id:
                return False, "Permission denied"
            
            # Delete report
            cursor.execute("""
            DELETE FROM report_history WHERE id = ?
            """, (report_id,))
            
            conn.commit()
            conn.close()
            return True, "Report deleted"
        
        except Exception as e:
            return False, f"Error deleting report: {str(e)}"

import threading
import time
import os
import logging
from datetime import datetime, timedelta
from modules.bedrock_integration import BedrockClient

# Configure logging for warmup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('warmup_scheduler')

class ModelWarmupScheduler:
    def __init__(self, warmup_interval_minutes=15):
        """
        Initialize the warmup scheduler
        
        Args:
            warmup_interval_minutes (int): How often to send warmup requests (default: 15 minutes)
        """
        self.warmup_interval = warmup_interval_minutes * 60  # Convert to seconds
        self.bedrock_client = BedrockClient()
        self.model_id = os.environ.get('BEDROCK_MODEL_ID', 'mistral.mistral-8b-instruct-v1:0')
        self.is_running = False
        self.warmup_thread = None
        self.last_real_request = datetime.now()
        self.warmup_stats = {
            'total_warmups': 0,
            'successful_warmups': 0,
            'failed_warmups': 0,
            'last_warmup': None,
            'last_warmup_success': None
        }
        
    def update_last_request_time(self):
        """Call this whenever a real user request is processed"""
        self.last_real_request = datetime.now()
        logger.debug(f"Updated last real request time: {self.last_real_request}")
        
    def should_warmup(self):
        """
        Determine if we should send a warmup request
        Returns True if last real request was more than warmup_interval ago
        """
        time_since_last_request = datetime.now() - self.last_real_request
        should_warmup = time_since_last_request.total_seconds() > (self.warmup_interval * 0.8)  # 80% of interval
        return should_warmup
        
    def send_warmup_request(self):
        """Send a minimal warmup request to keep the model alive"""
        try:
            # Minimal warmup prompt - only 3-4 tokens total
            warmup_prompt = "<s>[INST] Hi [/INST]"
            
            logger.info("🔥 Sending warmup request to Bedrock...")
            start_time = time.time()
            
            # Send warmup request using the configured model ID
            response = self.bedrock_client._call_bedrock_with_retry(
                model_id=self.model_id,  # Use the configured model ID
                prompt=warmup_prompt,
                max_tokens=5,  # Minimal response
                temperature=0.1
            )
            
            response_time = time.time() - start_time
            
            # Update stats
            self.warmup_stats['total_warmups'] += 1
            self.warmup_stats['successful_warmups'] += 1
            self.warmup_stats['last_warmup'] = datetime.now()
            self.warmup_stats['last_warmup_success'] = True
            
            logger.info(f"✅ Warmup successful in {response_time:.2f}s - Model is warm and ready")
            return True
            
        except Exception as e:
            self.warmup_stats['total_warmups'] += 1
            self.warmup_stats['failed_warmups'] += 1
            self.warmup_stats['last_warmup'] = datetime.now()
            self.warmup_stats['last_warmup_success'] = False
            
            logger.error(f"❌ Warmup failed: {str(e)}")
            return False
            
    def warmup_worker(self):
        """Background thread that handles periodic warmup"""
        logger.info(f"⏰ Warmup scheduler started - periodic warmup every {self.warmup_interval/60:.1f} minutes")
        
        # Small initial delay to avoid immediate warmup after startup warmup
        time.sleep(30)  # Wait 30 seconds before starting periodic schedule
        
        while self.is_running:
            try:
                # Check if we need to warmup
                if self.should_warmup():
                    self.send_warmup_request()
                else:
                    logger.debug("⏭️ Skipping warmup - recent user activity detected")
                
                # Sleep for the warmup interval
                time.sleep(self.warmup_interval)
                
            except Exception as e:
                logger.error(f"Error in warmup worker: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying on error
                
    def start(self):
        """Start the warmup scheduler"""
        if self.is_running:
            logger.warning("Warmup scheduler is already running")
            return
            
        self.is_running = True
        
        # Send initial warmup request immediately on startup
        logger.info("🚀 Sending initial warmup request on startup...")
        initial_success = self.send_warmup_request()
        if initial_success:
            logger.info("✅ Initial warmup successful - model is ready!")
        else:
            logger.warning("⚠️ Initial warmup failed - will retry on schedule")
        
        self.warmup_thread = threading.Thread(target=self.warmup_worker, daemon=True)
        self.warmup_thread.start()
        logger.info("✅ Warmup scheduler started successfully")
        
    def stop(self):
        """Stop the warmup scheduler"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.warmup_thread:
            self.warmup_thread.join(timeout=5)
        logger.info("🛑 Warmup scheduler stopped")
        
    def get_stats(self):
        """Get warmup statistics"""
        stats = self.warmup_stats.copy()
        
        # Add computed stats
        if stats['total_warmups'] > 0:
            stats['success_rate'] = (stats['successful_warmups'] / stats['total_warmups']) * 100
        else:
            stats['success_rate'] = 0
            
        # Add time since last request
        stats['minutes_since_last_request'] = (datetime.now() - self.last_real_request).total_seconds() / 60
        
        # Add next warmup estimate
        if self.should_warmup():
            stats['next_warmup'] = "Soon (model likely cold)"
        else:
            next_warmup_time = self.last_real_request + timedelta(seconds=self.warmup_interval * 0.8)
            stats['next_warmup'] = next_warmup_time.strftime("%H:%M:%S")
            
        return stats

# Global warmup scheduler instance
warmup_scheduler = None

def initialize_warmup_scheduler(warmup_interval_minutes=15):
    """Initialize and start the global warmup scheduler"""
    global warmup_scheduler
    
    if warmup_scheduler is None:
        warmup_scheduler = ModelWarmupScheduler(warmup_interval_minutes)
        warmup_scheduler.start()
        logger.info(f"🔥 Global warmup scheduler initialized with {warmup_interval_minutes} minute intervals")
    
    return warmup_scheduler

def get_warmup_scheduler():
    """Get the global warmup scheduler instance"""
    return warmup_scheduler

def update_last_request_time():
    """Update the last request time (call this on every real user request)"""
    if warmup_scheduler:
        warmup_scheduler.update_last_request_time() 
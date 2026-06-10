"""
Supervisor Event Listener
Monitors process state changes and logs events
"""

import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_stdout(s):
    """Write to stdout for supervisor"""
    sys.stdout.write(s)
    sys.stdout.flush()

def write_stderr(s):
    """Write to stderr"""
    sys.stderr.write(s)
    sys.stderr.flush()

def main():
    """Main event listener loop"""
    logger.info("Supervisor event listener started")
    
    while True:
        # Signal supervisor we're ready
        write_stdout('READY\n')
        
        # Read header line
        line = sys.stdin.readline()
        headers = dict([x.split(':') for x in line.split()])
        
        # Read event data
        data = sys.stdin.read(int(headers['len']))
        
        # Parse event
        event_data = dict([x.split(':') for x in data.split()])
        
        # Log event
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        process_name = event_data.get('processname', 'unknown')
        event_name = headers.get('eventname', 'unknown')
        
        logger.info(f"[{timestamp}] {event_name}: {process_name}")
        
        # Handle specific events
        if 'PROCESS_STATE_FATAL' in event_name:
            logger.error(f"FATAL: {process_name} crashed!")
        elif 'PROCESS_STATE_STOPPED' in event_name:
            logger.warning(f"STOPPED: {process_name}")
        elif 'PROCESS_STATE_STARTING' in event_name:
            logger.info(f"STARTING: {process_name}")
        elif 'PROCESS_STATE_RUNNING' in event_name:
            logger.info(f"RUNNING: {process_name}")
        
        # Acknowledge event
        write_stdout('RESULT 2\nOK')

if __name__ == '__main__':
    main()

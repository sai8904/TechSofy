import requests
import json
import os
from datetime import datetime

def send_slack_alert(webhook_url, alert_data):
    """Send alert to Slack webhook"""
    message = {
        "text": f"🚨 Health Orchestrator Alert",
        "attachments": [
            {
                "color": "danger" if alert_data.get('severity') == 'critical' else "warning",
                "fields": [
                    {
                        "title": "Service",
                        "value": alert_data.get('service', 'Unknown'),
                        "short": True
                    },
                    {
                        "title": "Alert Type",
                        "value": alert_data.get('type', 'Unknown'),
                        "short": True
                    },
                    {
                        "title": "Severity",
                        "value": alert_data.get('severity', 'Unknown'),
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": datetime.now().isoformat(),
                        "short": True
                    }
                ]
            }
        ]
    }
    
    response = requests.post(webhook_url, json=message)
    return response.status_code == 200

def send_email_alert(smtp_config, alert_data):
    """Send alert via email"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    msg = MIMEMultipart()
    msg['From'] = smtp_config['username']
    msg['To'] = smtp_config['to']
    msg['Subject'] = f"Health Orchestrator Alert: {alert_data.get('type', 'Unknown')}"
    
    body = f"""
    Alert Details:
    - Service: {alert_data.get('service', 'Unknown')}
    - Type: {alert_data.get('type', 'Unknown')}
    - Severity: {alert_data.get('severity', 'Unknown')}
    - Timestamp: {datetime.now().isoformat()}
    - Description: {alert_data.get('description', 'No description')}
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP(smtp_config['server'], smtp_config['port'])
    server.starttls()
    server.login(smtp_config['username'], smtp_config['password'])
    server.send_message(msg)
    server.quit()

if __name__ == "__main__":
    # Example usage
    alert_data = {
        "service": "user-service",
        "type": "high_error_rate",
        "severity": "warning",
        "description": "Error rate exceeded 10%"
    }
    
    webhook_url = os.getenv('ALERT_WEBHOOK_URL')
    if webhook_url:
        send_slack_alert(webhook_url, alert_data)
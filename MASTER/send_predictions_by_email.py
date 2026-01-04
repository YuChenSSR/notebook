import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def send_email_via_126(
    sender_email: str, 
    sender_auth_code: str,  # 注意：必须使用授权码，非邮箱密码！
    receiver_email: str,
    csv_file_path: str,
    subject: str = "CSV文件",
    body: str = "请查收附件中的CSV文件"
):
    """
    通过126邮箱发送带CSV附件的邮件
    """
    
    # ----------------------
    # 关键修改点：126邮箱SMTP配置
    # ----------------------
    smtp_server = "smtp.126.com"  # 服务器地址变更
    smtp_port = 465               # SSL加密端口（126邮箱必须使用SSL）
    
    # 创建邮件对象（与之前相同）
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    # 添加附件（代码无需修改）
    try:
        with open(csv_file_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{csv_file_path.split("/")[-1]}"')
        msg.attach(part)
    except Exception as e:
        raise Exception(f"附件添加失败: {str(e)}")
    
    # 发送邮件（注意SMTP_SSL）
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_auth_code)  # 使用授权码登录
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("邮件发送成功！")
    except Exception as e:
        raise Exception(f"发送失败: {str(e)}")
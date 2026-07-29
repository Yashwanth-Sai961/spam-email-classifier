from email import policy
from email.parser import BytesParser


def parse_eml_file(file_path):
    """
    Parse a .eml file and extract sender, subject, body, and attachments.
    """

    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    sender = msg.get("From", "")
    subject = msg.get("Subject", "")

    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()

            if content_type == "text/plain" and disposition != "attachment":
                try:
                    body = part.get_content()
                    break
                except Exception:
                    pass
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = ""

    attachments = []

    for part in msg.walk():
        if part.get_content_disposition() == "attachment":
            filename = part.get_filename()
            if filename:
                attachments.append(filename)

    return {
        "sender": sender,
        "subject": subject,
        "body": body,
        "attachments": attachments,
    }


if __name__ == "__main__":
    try:
        email_data = parse_eml_file("test_emails/malicious_test.eml")

        print("\n===== EMAIL PARSED SUCCESSFULLY =====")
        print("Sender:", email_data["sender"])
        print("Subject:", email_data["subject"])
        print("Body:")
        print(email_data["body"])
        print("Attachments:", email_data["attachments"])

    except Exception as e:
        print("Error:", e)
def analyze_email_file(file_path):
    """
    Analyze a .eml file.
    """

    email_data = parse_eml_file(file_path)

    prediction = predict(email_data["body"])

    attachment_result = check_attachments(email_data["attachments"])

    return {
        "sender": email_data["sender"],
        "subject": email_data["subject"],
        "prediction": prediction,
        "attachments": attachment_result
    }
if __name__ == "__main__":
    result = analyze_email_file("test_emails/malicious_test.eml")

    print(result)
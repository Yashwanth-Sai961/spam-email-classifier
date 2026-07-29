DANGEROUS_EXTENSIONS = {
    ".exe",
    ".bat",
    ".cmd",
    ".scr",
    ".js",
    ".vbs",
    ".msi",
    ".jar",
    ".ps1",
    ".dll"
}


def check_attachments(attachments):
    results = []
    risk = False

    for filename in attachments:
        dangerous = any(filename.lower().endswith(ext) for ext in DANGEROUS_EXTENSIONS)

        if dangerous:
            risk = True

        results.append({
            "filename": filename,
            "dangerous": dangerous
        })

    return {
        "attachment_risk": risk,
        "attachments": results
    }
if __name__ == "__main__":
    sample = [
        "invoice.pdf",
        "setup.exe",
        "photo.jpg",
        "script.js"
    ]

    result = check_attachments(sample)

    print(result)
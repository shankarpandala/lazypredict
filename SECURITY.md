# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take the security of LazyPredict seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to shankar.pandala@live.com with the subject line "LazyPredict Security Vulnerability".

Please include the following information in your report:

* Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, we will:

1. **Acknowledge receipt** of your vulnerability report within 48 hours
2. **Confirm the problem** and determine the affected versions within 5 business days
3. **Audit code** to find any similar problems
4. **Prepare fixes** for all supported versions
5. **Release new security fix versions** as soon as possible
6. **Publicly disclose** the vulnerability after the fix is released

### Disclosure Policy

When we receive a security bug report, we will:

* Confirm the problem and determine affected versions
* Audit code to find any similar problems
* Prepare fixes for all still-supported versions
* Release new versions as soon as possible
* Publicly acknowledge the reporter (unless they wish to remain anonymous)

### Security Update Notifications

Security updates will be announced through:

* GitHub Security Advisories
* Release notes on GitHub
* PyPI package changelog

### Safe Harbor

We support safe harbor for security researchers who:

* Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of our services
* Only interact with accounts you own or with explicit permission of the account holder
* Do not exploit a security issue for any reason (this includes demonstrating additional risk)
* Give us a reasonable time to resolve the issue before any disclosure to the public or a third party

### Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request or send an email to shankar.pandala@live.com.

## Security Best Practices for Users

When using LazyPredict, we recommend:

1. **Keep dependencies updated**: Regularly update LazyPredict and all dependencies
2. **Use virtual environments**: Isolate your LazyPredict installation
3. **Validate custom metrics**: If using custom metric functions, ensure they are from trusted sources
4. **Secure MLflow tracking**: If using MLflow with remote tracking, use proper authentication
5. **Review model outputs**: Always validate model predictions before use in production

## Known Security Considerations

### Custom Metrics

LazyPredict accepts custom metric functions via the `custom_metric` parameter. Users should only pass trusted callables, as these functions are executed during model evaluation.

### MLflow Integration

When using MLflow tracking, be aware that:
* The tracking URI can be set via environment variables
* Proper authentication should be used for remote tracking servers
* Experiment data may contain sensitive information

### Dependency Chain

LazyPredict depends on several machine learning libraries. Security updates to these dependencies will trigger updates to LazyPredict when necessary.

## Acknowledgments

We would like to publicly thank the following individuals for responsibly disclosing security vulnerabilities:

* (No vulnerabilities reported yet)

---

Last updated: October 5, 2025

#!/usr/bin/env python3
"""
GitHub Repository Setup Script for Cursey
Automates the process of creating and publishing to GitHub

Author: Cursey Team
Date: 2025
"""

import os
import sys
import subprocess
import json
from pathlib import Path


class GitHubSetup:
    """GitHub repository setup and publishing automation."""
    
    def __init__(self, repo_name="cursey", username=None, description=None):
        self.repo_name = repo_name
        self.username = username
        self.description = description or "Multi-Person Face & Eye Tracking System with Detectron2"
        self.repo_path = Path.cwd()
        
    def check_git_config(self):
        """Check if Git is configured properly."""
        print("Checking Git configuration...")
        
        try:
            # Check if git is installed
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            
            # Check user configuration
            result = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True)
            if not result.stdout.strip():
                print("⚠ Git user.name not configured")
                return False
                
            result = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True)
            if not result.stdout.strip():
                print("⚠ Git user.email not configured")
                return False
                
            print("✓ Git configuration looks good")
            return True
            
        except subprocess.CalledProcessError:
            print("✗ Git is not installed or not in PATH")
            return False
    
    def check_github_cli(self):
        """Check if GitHub CLI is installed."""
        print("Checking GitHub CLI...")
        
        try:
            result = subprocess.run(["gh", "--version"], check=True, capture_output=True, text=True)
            print("✓ GitHub CLI is installed")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ GitHub CLI not found")
            print("Install it from: https://cli.github.com/")
            return False
    
    def create_github_repo(self):
        """Create GitHub repository using GitHub CLI."""
        print(f"Creating GitHub repository: {self.username}/{self.repo_name}")
        
        try:
            cmd = [
                "gh", "repo", "create", self.repo_name,
                "--description", self.description,
                "--public",
                "--source", ".",
                "--remote", "origin",
                "--push"
            ]
            
            if self.username:
                cmd.extend(["--owner", self.username])
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ GitHub repository created successfully")
            print(f"Repository URL: https://github.com/{self.username}/{self.repo_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create repository: {e}")
            print("Error output:", e.stderr)
            return False
    
    def setup_remote_manual(self):
        """Setup remote repository manually."""
        print("Setting up remote repository manually...")
        
        if not self.username:
            self.username = input("Enter your GitHub username: ").strip()
        
        repo_url = f"https://github.com/{self.username}/{self.repo_name}.git"
        
        try:
            # Add remote origin
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
            print(f"✓ Remote origin added: {repo_url}")
            
            # Push to GitHub
            subprocess.run(["git", "branch", "-M", "main"], check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
            print("✓ Code pushed to GitHub successfully")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to setup remote: {e}")
            return False
    
    def create_release(self, version="2.0.0"):
        """Create a GitHub release."""
        print(f"Creating release {version}...")
        
        try:
            # Create tag
            subprocess.run(["git", "tag", "-a", f"v{version}", "-m", f"Release {version}"], check=True)
            subprocess.run(["git", "push", "origin", f"v{version}"], check=True)
            
            # Create release using GitHub CLI
            cmd = [
                "gh", "release", "create", f"v{version}",
                "--title", f"Cursey {version}",
                "--notes", f"""
# Cursey {version} Release

## 🚀 New Features
- Facebook Detectron2 integration for state-of-the-art detection
- Modern neumorphism UI with clean design
- GPU acceleration support with CPU fallback
- Comprehensive test suite and CI/CD pipeline

## 📦 Installation
```bash
pip install cursey
```

## 🎯 Quick Start
```bash
python -m cursey.examples.detectron_main_app
```

## 📚 Documentation
See README.md for complete documentation and examples.

## 🤝 Contributing
We welcome contributions! See CONTRIBUTING.md for guidelines.
                """.strip()
            ]
            
            subprocess.run(cmd, check=True)
            print(f"✓ Release {version} created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create release: {e}")
            return False
    
    def setup_github_pages(self):
        """Setup GitHub Pages for documentation."""
        print("Setting up GitHub Pages...")
        
        try:
            # Enable GitHub Pages
            cmd = [
                "gh", "api", "repos", f"{self.username}/{self.repo_name}", "pages",
                "--method", "POST",
                "--field", "source[branch]=main",
                "--field", "source[path]=/docs"
            ]
            
            subprocess.run(cmd, check=True)
            print("✓ GitHub Pages enabled")
            print(f"Documentation will be available at: https://{self.username}.github.io/{self.repo_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to setup GitHub Pages: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process."""
        print("Cursey GitHub Repository Setup")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_git_config():
            print("Please configure Git first:")
            print("git config --global user.name 'Your Name'")
            print("git config --global user.email 'your.email@example.com'")
            return False
        
        # Check if repository already exists
        if os.path.exists(".git"):
            print("✓ Git repository already initialized")
        else:
            print("✗ Git repository not found")
            return False
        
        # Try GitHub CLI first
        if self.check_github_cli():
            if self.create_github_repo():
                print("\n🎉 Repository created successfully!")
                
                # Ask about creating release
                create_release = input("Create initial release? (y/n): ").lower().strip()
                if create_release in ['y', 'yes']:
                    self.create_release()
                
                # Ask about GitHub Pages
                setup_pages = input("Setup GitHub Pages for documentation? (y/n): ").lower().strip()
                if setup_pages in ['y', 'yes']:
                    self.setup_github_pages()
                
                return True
        
        # Fallback to manual setup
        print("\nManual setup required:")
        print("1. Go to https://github.com/new")
        print(f"2. Create repository named '{self.repo_name}'")
        print("3. Run this script again")
        
        return False


def main():
    """Main setup function."""
    print("Cursey GitHub Repository Setup")
    print("=" * 50)
    
    # Get repository details
    repo_name = input("Repository name (default: cursey): ").strip() or "cursey"
    username = input("GitHub username: ").strip()
    description = input("Repository description (optional): ").strip()
    
    # Create setup instance
    setup = GitHubSetup(repo_name, username, description)
    
    # Run setup
    success = setup.run_setup()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print(f"Your repository is ready at: https://github.com/{username}/{repo_name}")
    else:
        print("\n⚠ Setup incomplete. Please follow the manual instructions above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

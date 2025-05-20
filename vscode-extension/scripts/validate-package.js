#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const util = require('util');

const execAsync = util.promisify(exec);
console.log('Starting package validation...');

/**
 * Validates required fields in package.json
 */
function validatePackageJson() {
  try {
    const packageJsonPath = path.join(process.cwd(), 'package.json');
    console.log(`Reading package.json from ${packageJsonPath}`);
    const packageJsonContent = fs.readFileSync(packageJsonPath, 'utf8');
    console.log('Package.json content loaded');
    const packageJson = JSON.parse(packageJsonContent);

    // Required fields
    const requiredFields = [
      'name', 'displayName', 'description', 'version', 
      'publisher', 'engines', 'license', 'repository'
    ];

    const missingFields = requiredFields.filter(field => !packageJson[field]);
    
    if (missingFields.length > 0) {
      console.error('Missing required fields:', missingFields.join(', '));
      process.exit(1);
    }
    
    // Validate icon exists
    if (packageJson.icon) {
      const iconPath = path.join(process.cwd(), packageJson.icon);
      if (!fs.existsSync(iconPath)) {
        console.error(`Icon file not found: ${packageJson.icon}`);
        process.exit(1);
      }
    } else {
      console.warn('Warning: No icon specified in package.json');
    }
    
    // Check README.md
    const readmePath = path.join(process.cwd(), 'README.md');
    if (!fs.existsSync(readmePath)) {
      console.error('README.md is missing');
      process.exit(1);
    }
    
    // Check CHANGELOG.md
    const changelogPath = path.join(process.cwd(), 'CHANGELOG.md');
    if (!fs.existsSync(changelogPath)) {
      console.error('CHANGELOG.md is missing');
      process.exit(1);
    }
    
    console.log('package.json validation passed!');
    return true;
  } catch (err) {
    console.error('Error validating package.json:', err);
    process.exit(1);
  }
}

/**
 * Run vsce package command
 */
async function runVscePackage() {
  try {
    console.log('Running vsce package...');
    const { stdout, stderr } = await execAsync('npx @vscode/vsce package');
    console.log(stdout);
    if (stderr) {
      console.error(stderr);
    }
    
    // Verify .vsix file exists
    const packageJsonPath = path.join(process.cwd(), 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    const expectedVsixName = `${packageJson.name}-${packageJson.version}.vsix`;
    
    if (fs.existsSync(path.join(process.cwd(), expectedVsixName))) {
      console.log(`Success! Created: ${expectedVsixName}`);
      return true;
    } else {
      console.error('Failed to find generated .vsix file');
      return false;
    }
  } catch (err) {
    console.error('Error packaging extension:', err);
    return false;
  }
}

async function main() {
  const isValid = validatePackageJson();
  if (isValid) {
    await runVscePackage();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

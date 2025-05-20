import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import MainPanel from '../src/components/MainPanel';
import Navigation from '../src/components/Navigation';

describe('MainPanel', () => {
  it('renders the main panel container', () => {
    render(<MainPanel />);
    expect(screen.getByTestId('main-panel')).toBeInTheDocument();
  });

  it('shows the navigation component', () => {
    render(<MainPanel />);
    expect(screen.getByTestId('navigation')).toBeInTheDocument();
  });

  it('displays the default view (e.g., upload dataset)', () => {
    render(<MainPanel />);
    expect(screen.getByText(/upload dataset/i)).toBeInTheDocument();
  });
});

describe('Navigation', () => {
  it('renders navigation links', () => {
    render(<Navigation />);
    expect(screen.getByText(/upload/i)).toBeInTheDocument();
    expect(screen.getByText(/models/i)).toBeInTheDocument();
    expect(screen.getByText(/results/i)).toBeInTheDocument();
  });

  it('highlights the selected section', () => {
    render(<Navigation selected="models" />);
    expect(screen.getByText(/models/i)).toHaveClass('selected');
  });
});

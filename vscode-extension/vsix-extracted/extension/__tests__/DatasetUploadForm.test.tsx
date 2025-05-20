import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import DatasetUploadForm from '../src/components/DatasetUploadForm';

describe('DatasetUploadForm', () => {
  it('renders the upload form', () => {
    render(<DatasetUploadForm />);
    expect(screen.getByTestId('dataset-upload-form')).toBeInTheDocument();
  });

  it('has a file input for dataset upload', () => {
    render(<DatasetUploadForm />);
    expect(screen.getByLabelText(/choose dataset file/i)).toBeInTheDocument();
  });

  it('has a submit button', () => {
    render(<DatasetUploadForm />);
    expect(screen.getByRole('button', { name: /upload/i })).toBeInTheDocument();
  });

  it('calls onUpload with the selected file when submitted', () => {
    const handleUpload = jest.fn();
    render(<DatasetUploadForm onUpload={handleUpload} />);
    const file = new File(['dummy content'], 'test.csv', { type: 'text/csv' });
    const input = screen.getByLabelText(/choose dataset file/i);
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByRole('button', { name: /upload/i }));
    expect(handleUpload).toHaveBeenCalledWith(file);
  });
});

import React, { useState } from 'react';

interface DatasetUploadFormProps {
  onUpload?: (file: File) => void;
}

const DatasetUploadForm: React.FC<DatasetUploadFormProps> = ({ onUpload }) => {
  const [file, setFile] = useState<File | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (file && onUpload) {
      onUpload(file);
    }
  };

  return (
    <form data-testid="dataset-upload-form" onSubmit={handleSubmit}>
      <label htmlFor="dataset-upload-input">Choose dataset file</label>
      <input
        id="dataset-upload-input"
        type="file"
        accept=".csv,.xlsx,.xls"
        onChange={handleChange}
        aria-label="Choose dataset file"
      />
      <button type="submit">Upload</button>
    </form>
  );
};

export default DatasetUploadForm;

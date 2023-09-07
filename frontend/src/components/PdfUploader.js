import React, { useState } from 'react';

function PdfUploader({ onUpload }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);


  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (file) {
      setLoading(true); // Start loading
  
      try {
        await onUpload(file);
      } catch (error) {
        console.error('API call error:', error);
      } finally {
        setLoading(false); // Stop loading
      }
    }
  };

  return (
    <div class="header">
      <input type="file" accept=".pdf" onChange={handleFileChange} />
      <br /><br />
      <button className="button" onClick={handleUpload}>
        {loading ? 'Uploading...' : 'Upload'}
      </button>
      {loading && <div className="loading-spinner"></div>} {/* Add loading spinner */}
    </div>
  );
}

export default PdfUploader;

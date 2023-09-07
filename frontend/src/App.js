import React, { useState } from 'react';
import PdfUploader from './components/PdfUploader';
import {v4 as uuidv4} from 'uuid';



function App() {
  const [uploads, setUploads] = useState([]);

  const handleUpload = async (file) => {
    const formData = new FormData();
    const currentDate = new Date().toJSON().slice(0, 10);
    const myuuid = uuidv4();
    formData.append('file', file);
  
    const response = await fetch(`/${myuuid}_${currentDate}/upload`, {
      method: 'POST',
      body: formData,
    });
  
    const data = await response.json();
    console.log(data);
    // Add the new upload to the state
    setUploads([...uploads, data]);
  };


  return (
    <div class='main'>
      <h1> Pharos</h1>
      <p> Welcome to Pharos! This tool allows you to easily upload film scripts and automatically generate summaries. </p>
      <PdfUploader onUpload={handleUpload} />
      <div>
        {uploads.map((upload, index) => (
          <div key={index}>
            <p class="script">{upload.script}</p>
            <p class="summary">Summary: {upload.summary}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
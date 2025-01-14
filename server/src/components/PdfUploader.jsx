import React, { useState } from 'react';

const PdfUploader = () => {
    const [selectedFile, setSelectedFile] = useState(null);

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        if (selectedFile) {
            console.log('File selected:', selectedFile.name);
            alert(`File "${selectedFile.name}" uploaded successfully!`);
        } else {
            alert('No file selected!');
        }
    };

    return (
        <div className="file-uploader" style={{ textAlign: 'center', padding: '20px' }}>
            <form onSubmit={handleSubmit}>
                <input
                    type="file"
                    onChange={handleFileChange}
                    style={{
                        margin: '10px 0',
                        padding: '10px',
                        borderRadius: '5px',
                        border: '1px solid #ccc',
                    }}
                />
                <br />
                <button
                    type="submit"
                    style={{
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        padding: '10px 20px',
                        borderRadius: '5px',
                        cursor: 'pointer',
                    }}
                >
                    Upload
                </button>
            </form>
            {selectedFile && (
                <p style={{ marginTop: '10px', color: '#555' }}>
                    Selected File: <strong>{selectedFile.name}</strong>
                </p>
            )}
        </div>
    );
};

export default PdfUploader;

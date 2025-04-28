import React, { useState } from 'react';

export default function Landing() {
  const [recognizedText, setRecognizedText] = useState("");

  const handleRecognize = async () => {
    try {
      const response = await fetch("http://localhost:8000/capture-and-recognize");
      const data = await response.json();
      setRecognizedText(data.recognized);
    } catch (error) {
      console.error("Error recognizing:", error);
    }
  };

  return (
    <div className="landing">
      <main className="landing-hero container">
        <h2>Welcome to Skywriter</h2>
        <p>Capture, organize and access your notes anywhere.</p>

        {/* Live canvas feed from backend */}
        <div style={{ margin: "20px 0" }}>
          <img
            src="http://localhost:8000/video_feed"
            alt="Drawing Canvas"
            style={{ width: "640px", height: "480px", border: "2px solid black", borderRadius: "10px" }}
          />
        </div>

        {/* Button to trigger recognition */}
        <button onClick={handleRecognize} style={{ marginBottom: "20px" }}>
          Recognize
        </button>

        {/* Output */}
        {recognizedText && (
          <div>
            <h3>Recognized Text:</h3>
            <p style={{ fontSize: "24px", fontWeight: "bold" }}>{recognizedText}</p>
          </div>
        )}
      </main>
    </div>
  );
}

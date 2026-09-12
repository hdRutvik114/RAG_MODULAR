import { useState } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [pdfId, setPdfId] = useState('');
  const [question, setQuestion] = useState('What is this document about?');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [uploadMessage, setUploadMessage] = useState('');
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file) {
      setError('Please choose a PDF file first.');
      return;
    }

    setUploading(true);
    setError('');
    setUploadMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed.');
      }

      setPdfId(data.pdf_id);
      setUploadMessage(`Uploaded successfully. PDF ID: ${data.pdf_id}`);
    } catch (err) {
      setError(err.message || 'Could not upload the PDF.');
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!pdfId || !question.trim()) {
      setError('Upload a PDF and enter a question before querying.');
      return;
    }

    setQuerying(true);
    setError('');

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pdf_id: pdfId, question }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Query failed.');
      }

      setAnswer(data.answer || 'No answer returned.');
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message || 'Query failed.');
    } finally {
      setQuerying(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="card">
        <h1>RAG PDF Assistant</h1>

        <div className="section">
          <label htmlFor="pdf-upload">Upload a PDF</label>
          <input
            id="pdf-upload"
            type="file"
            accept="application/pdf"
            onChange={(event) => setFile(event.target.files?.[0] || null)}
          />
          <button onClick={handleUpload} disabled={uploading || !file}>
            {uploading ? 'Uploading...' : 'Upload PDF'}
          </button>
        </div>

        {uploadMessage && <p className="success">{uploadMessage}</p>}

        <div className="section">
          <label htmlFor="pdf-id">PDF ID</label>
          <input
            id="pdf-id"
            value={pdfId}
            onChange={(event) => setPdfId(event.target.value)}
            placeholder="The PDF ID returned by the backend"
          />
        </div>

        <div className="section">
          <label htmlFor="question">Ask a question</label>
          <textarea
            id="question"
            rows={4}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button onClick={handleQuery} disabled={querying || !pdfId}>
            {querying ? 'Searching...' : 'Ask question'}
          </button>
        </div>

        {error && <p className="error">{error}</p>}

        {answer && (
          <div className="result-box">
            <h2>Answer</h2>
            <p>{answer}</p>

            {sources.length > 0 && (
              <div>
                <h3>Sources</h3>
                <ul>
                  {sources.map((source, index) => (
                    <li key={`${source.text}-${index}`}>
                      <strong>Score:</strong> {Number(source.score || 0).toFixed(4)}
                      <br />
                      {source.text.slice(0, 240)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

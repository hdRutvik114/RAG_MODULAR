import { useState, useEffect, useRef } from 'react';

export default function App() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  // activeDocument stores { id: string, name: string }
  const [activeDoc, setActiveDoc] = useState(null);
  const [question, setQuestion] = useState('What are the main key points of this document?');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [error, setError] = useState('');
  const [uploadMessage, setUploadMessage] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking');

  const fileInputRef = useRef(null);
  const questionInputRef = useRef(null);

  // Check backend connectivity on load
  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (res.ok) setBackendStatus('connected');
        else setBackendStatus('error');
      })
      .catch(() => setBackendStatus('disconnected'));
  }, []);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
        setError('Please select a valid PDF file.');
        return;
      }
      setFile(selectedFile);
      setError('');
      // Auto trigger upload once selected
      uploadFile(selectedFile);
    }
  };

  const uploadFile = async (fileToUpload) => {
    setUploading(true);
    setError('');
    setUploadMessage('');
    setAnswer('');
    setSources([]);

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed.');
      }

      // Store active document automatically - user never needs to copy/paste IDs
      setActiveDoc({
        id: data.pdf_id,
        name: fileToUpload.name,
      });

      setUploadMessage(`"${fileToUpload.name}" indexed successfully!`);
      // Focus question box for instant querying
      setTimeout(() => questionInputRef.current?.focus(), 100);
    } catch (err) {
      setError(err.message || 'Could not upload and index the PDF.');
    } finally {
      setUploading(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    if (!activeDoc?.id) {
      setError('Please upload a PDF document first.');
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
        // Automatically pass activeDoc.id
        body: JSON.stringify({
          pdf_id: activeDoc.id,
          question: question.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Query failed.');
      }

      setAnswer(data.answer || 'No answer returned.');
      setSources(data.sources || []);
    } catch (err) {
      setError(err.message || 'Failed to get answer from the document.');
    } finally {
      setQuerying(false);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleQuery();
    }
  };

  const handleResetDocument = () => {
    setActiveDoc(null);
    setFile(null);
    setAnswer('');
    setSources([]);
    setUploadMessage('');
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-shell">
      <div className="card">
        {/* Header */}
        <header className="header">
          <div className="title-area">
            <h1>RAG PDF Assistant</h1>
            <p>Hybrid Search + AI Document Q&A</p>
          </div>
          <div className={`status-badge ${backendStatus === 'connected' ? 'status-connected' : 'status-error'}`}>
            <span className="status-dot"></span>
            Backend: {backendStatus}
          </div>
        </header>

        {/* Upload Section */}
        {!activeDoc ? (
          <div
            className="dropzone"
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              className="file-input"
              onChange={handleFileChange}
            />
            <div className="dropzone-content">
              <span className="upload-icon">📄</span>
              <strong style={{ fontSize: '15px' }}>
                {uploading ? 'Processing & Indexing PDF...' : 'Click to Upload PDF Document'}
              </strong>
              <span style={{ fontSize: '12.5px', color: '#94a3b8' }}>
                {uploading ? 'Generating embeddings & BM25 indices...' : 'PDF files will be automatically indexed and ready for questions'}
              </span>
            </div>
          </div>
        ) : (
          /* Active Document Banner - Automatic State */
          <div className="active-doc-card">
            <div className="doc-info">
              <span className="doc-icon">📕</span>
              <div className="doc-details">
                <h4>{activeDoc.name}</h4>
                <p>✓ Indexed & Active for questions</p>
              </div>
            </div>
            <button
              className="change-btn"
              onClick={handleResetDocument}
              title="Upload a different PDF"
            >
              Upload Different PDF
            </button>
          </div>
        )}

        {uploadMessage && !error && (
          <div className="alert-success">✓ {uploadMessage}</div>
        )}

        {/* Question Area */}
        <div className="query-section">
          <label className="query-label" htmlFor="question-input">
            Ask a Question about the Document
          </label>
          <div className="query-input-wrapper">
            <textarea
              id="question-input"
              ref={questionInputRef}
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                activeDoc
                  ? `Ask anything about "${activeDoc.name}"...`
                  : 'Upload a PDF above, then ask your question here...'
              }
            />
          </div>

          <div className="action-row">
            <span className="shortcut-tip">Tip: Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to ask</span>
            <button
              className="ask-btn"
              onClick={handleQuery}
              disabled={querying || !activeDoc || !question.trim()}
            >
              {querying ? (
                <>
                  <span>Searching & Generating...</span>
                </>
              ) : (
                <>
                  <span>Ask Question</span>
                  <span>→</span>
                </>
              )}
            </button>
          </div>
        </div>

        {error && <div className="alert-error">⚠️ {error}</div>}

        {/* Answer & Sources Result */}
        {answer && (
          <div className="result-card">
            <div className="result-header">
              <span>🤖</span>
              <h3>Generated Answer</h3>
            </div>
            <div className="answer-body">{answer}</div>

            {sources.length > 0 && (
              <div className="sources-container">
                <div className="sources-title">Retrieved Evidence & Citations</div>
                {sources.map((source, index) => {
                  const score = source.rrf_score ?? source.score ?? 0;
                  const page = source.metadata?.page;
                  return (
                    <div className="source-item" key={index}>
                      <div className="source-meta">
                        <span className="score-tag">
                          Match Score: {Number(score).toFixed(4)}
                        </span>
                        {page !== undefined && (
                          <span className="page-tag">Page {Number(page) + 1}</span>
                        )}
                      </div>
                      <p className="source-snippet">
                        {source.text || ''}
                      </p>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

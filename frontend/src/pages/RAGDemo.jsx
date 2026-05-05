/* RAGDemo.jsx */
import { useState } from "react"
import { api } from "../lib/api"
import { useModelStore } from "../store/modelStore"

const QUESTIONS = [
  "How does TurboQuant compress KV cache?",
  "What is vector quantization used for in AI?",
  "How does attention mechanism work?",
  "What is the benefit of RAG over pure generation?",
]

export default function RAGDemo() {
  const { bits } = useModelStore()
  const [question, setQuestion] = useState(QUESTIONS[0])
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)

  async function run() {
    setLoading(true)
    const r = await api.rag({ question, bits })
    setResult(r)
    setLoading(false)
  }

  return (
    <div className="demo-page">
      <div className="demo-header">
        <h2>📚 RAG Pipeline</h2>
        <p>Retrieve compressed embeddings → pass context to loaded model → generate answer.</p>
      </div>
      <div className="controls-card">
        <div className="ctrl-group">
          <label>Question</label>
          <div className="preset-row">
            {QUESTIONS.map((q,i)=>(
              <button key={i} className={`preset-btn ${question===q?"on":""}`}
                onClick={()=>setQuestion(q)}>Q{i+1}</button>
            ))}
          </div>
          <input className="text-input" value={question} onChange={e=>setQuestion(e.target.value)} />
        </div>
        <button className="run-btn" onClick={run} disabled={loading}>
          {loading ? <><span className="spinner"/>Running RAG...</> : "▶  Run RAG"}
        </button>
      </div>

      {result && (
        <div className="rag-results">
          <div className="rag-retrieved">
            <h3>Retrieved Documents ({bits}-bit compressed)</h3>
            {result.retrieved_docs?.map((doc,i)=>(
              <div key={i} className="rag-doc">
                <span className="rag-doc-rank">#{i+1}</span>
                <p>{doc}</p>
              </div>
            ))}
          </div>
          <div className="rag-answer-box">
            <div className="rag-answer-header">
              Generated Answer
              <span className="badge">{result.compression_ratio}× compressed retrieval</span>
            </div>
            <p className="rag-answer-text">{result.answer}</p>
          </div>
        </div>
      )}
    </div>
  )
}

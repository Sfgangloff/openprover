import { useCallback, useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useProblem, useAdjacentProblems } from '../hooks/useData'
import MathMarkdown from '../components/MathMarkdown'
import CostBadge from '../components/CostBadge'
import '../components/MathMarkdown.css'
import './ProblemDetail.css'

function DownloadIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M8 2v8m0 0l-3-3m3 3l3-3M3 12h10" />
    </svg>
  )
}

export default function ProblemDetail() {
  const { paperId, problemId } = useParams()
  const navigate = useNavigate()
  const { paper, problem } = useProblem(paperId, problemId)
  const { prev, next, index, total } = useAdjacentProblems(paperId, problemId)
  const [raw, setRaw] = useState(false)

  const downloadPrompt = useCallback(() => {
    const parts = [`Prove the following conjecture.\n`]
    parts.push(`## ${problem.name}\n`)
    parts.push(problem.statement)
    if (problem.context) {
      parts.push(`\n\n### Context\n\n${problem.context}`)
    }
    if (problem.references && problem.references.length > 0) {
      parts.push(`\n\n### References\n`)
      problem.references.forEach(ref => {
        parts.push(`[${ref.tag}] ${ref.text}`)
      })
    }
    const blob = new Blob([parts.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${problem.name.replace(/[^a-zA-Z0-9]+/g, '_')}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [problem])

  // Keyboard navigation
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'ArrowLeft' && prev) {
        navigate(`/paper/${prev.paper.id}/problem/${prev.problem.id}`)
      } else if (e.key === 'ArrowRight' && next) {
        navigate(`/paper/${next.paper.id}/problem/${next.problem.id}`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prev, next, navigate])

  // Scroll to top on navigation
  useEffect(() => { window.scrollTo(0, 0) }, [problemId])

  if (!paper || !problem) {
    return <div className="problem-detail-empty">Problem not found.</div>
  }

  const ext = paper.extraction && !paper.extraction.error ? paper.extraction : null

  return (
    <div className="problem-detail">
      <h1 className="problem-detail-title">{problem.name}</h1>

      <div className="problem-detail-meta">
        {ext && <CostBadge cost={ext.cost} />}
        <span className="problem-detail-counter">{index + 1} / {total}</span>
      </div>

      {problem.summary && (
        <div className="problem-detail-summary">{problem.summary}</div>
      )}

      <section className="problem-detail-section">
        <div className="problem-detail-section-header">
          <h2>Statement</h2>
          <div className="problem-detail-actions">
            <button className="problem-detail-download" onClick={downloadPrompt} title="Download as prompt">
              <DownloadIcon /> Prompt
            </button>
            <label className="problem-detail-toggle">
              <span className="problem-detail-toggle-label">Raw</span>
              <input type="checkbox" checked={raw} onChange={() => setRaw(r => !r)} />
              <span className="problem-detail-toggle-track" />
            </label>
          </div>
        </div>
        {problem.location && <span className="problem-detail-loc">{problem.location}</span>}
        <div className="problem-detail-content">
          {raw
            ? <pre className="problem-detail-raw">{problem.statement}</pre>
            : <MathMarkdown key={problemId} references={problem.references}>{problem.statement}</MathMarkdown>}
        </div>
      </section>

      {problem.context && (
        <section className="problem-detail-section">
          <h2>Context</h2>
          <div className="problem-detail-content">
            {raw
              ? <pre className="problem-detail-raw">{problem.context}</pre>
              : <MathMarkdown key={`ctx-${problemId}`} references={problem.references}>{problem.context}</MathMarkdown>}
          </div>
        </section>
      )}

      {problem.references && problem.references.length > 0 && (
        <section className="problem-detail-section">
          <h2>References</h2>
          <ol className="problem-detail-refs">
            {problem.references.map((ref, i) => (
              <li key={i} id={`ref-${ref.tag}`}>
                <span className="problem-detail-ref-tag">[{ref.tag}]</span>{' '}
                <MathMarkdown key={`ref-${problemId}-${i}`} className="problem-detail-ref-text">{ref.text}</MathMarkdown>
              </li>
            ))}
          </ol>
        </section>
      )}

      <nav className="problem-nav">
        {prev ? (
          <Link to={`/paper/${prev.paper.id}/problem/${prev.problem.id}`} className="problem-nav-btn problem-nav-prev">
            <span className="problem-nav-arrow">&larr;</span>
            <span className="problem-nav-label">
              <span className="problem-nav-dir">Previous</span>
              <span className="problem-nav-name">{prev.problem.name}</span>
              {prev.paper.id !== paperId && (
                <span className="problem-nav-paper">{prev.paper.title}</span>
              )}
            </span>
          </Link>
        ) : <div />}

        {next ? (
          <Link to={`/paper/${next.paper.id}/problem/${next.problem.id}`} className="problem-nav-btn problem-nav-next">
            <span className="problem-nav-label">
              <span className="problem-nav-dir">Next</span>
              <span className="problem-nav-name">{next.problem.name}</span>
              {next.paper.id !== paperId && (
                <span className="problem-nav-paper">{next.paper.title}</span>
              )}
            </span>
            <span className="problem-nav-arrow">&rarr;</span>
          </Link>
        ) : <div />}
      </nav>
    </div>
  )
}

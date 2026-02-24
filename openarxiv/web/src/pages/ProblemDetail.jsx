import { useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { useProblem, useAdjacentProblems } from '../hooks/useData'
import MathMarkdown from '../components/MathMarkdown'
import CostBadge from '../components/CostBadge'
import '../components/MathMarkdown.css'
import './ProblemDetail.css'

export default function ProblemDetail() {
  const { paperId, problemId } = useParams()
  const { paper, problem } = useProblem(paperId, problemId)
  const { prev, next, index, total } = useAdjacentProblems(paperId, problemId)

  // Keyboard navigation
  useEffect(() => {
    function onKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (e.key === 'ArrowLeft' && prev) {
        window.location.href = `/paper/${prev.paper.id}/problem/${prev.problem.id}`
      } else if (e.key === 'ArrowRight' && next) {
        window.location.href = `/paper/${next.paper.id}/problem/${next.problem.id}`
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [prev, next])

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
        {problem.location && <span className="problem-detail-loc">{problem.location}</span>}
        {ext && <CostBadge cost={ext.cost} />}
        <span className="problem-detail-counter">{index + 1} / {total}</span>
      </div>

      {problem.summary && (
        <div className="problem-detail-summary">{problem.summary}</div>
      )}

      <section className="problem-detail-section">
        <h2>Statement</h2>
        <div className="problem-detail-content">
          <MathMarkdown references={problem.references}>{problem.statement}</MathMarkdown>
        </div>
      </section>

      {problem.context && (
        <section className="problem-detail-section">
          <h2>Context</h2>
          <div className="problem-detail-content">
            <MathMarkdown references={problem.references}>{problem.context}</MathMarkdown>
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
                <MathMarkdown className="problem-detail-ref-text">{ref.text}</MathMarkdown>
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

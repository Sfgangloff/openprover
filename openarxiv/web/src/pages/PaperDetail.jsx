import { useParams, Link } from 'react-router-dom'
import { usePaper } from '../hooks/useData'
import ProblemSummary from '../components/ProblemSummary'
import MathMarkdown from '../components/MathMarkdown'
import CostBadge from '../components/CostBadge'
import './PaperDetail.css'

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

export default function PaperDetail() {
  const { paperId } = useParams()
  const paper = usePaper(paperId)

  if (!paper) {
    return <div className="paper-detail-empty">Paper not found: {paperId}</div>
  }

  const ext = paper.extraction && !paper.extraction.error ? paper.extraction : null

  return (
    <div className="paper-detail">
      <h1 className="paper-detail-title">{paper.title}</h1>

      <div className="paper-detail-meta">
        <div className="paper-detail-authors">{paper.authors.join(', ')}</div>
        <div className="paper-detail-info">
          <span>{formatDate(paper.published)}</span>
          {paper.categories.map(cat => (
            <span key={cat} className="paper-detail-tag">{cat}</span>
          ))}
          <a href={paper.pdf_url} target="_blank" rel="noopener" className="paper-detail-pdf">PDF</a>
          <span className="paper-detail-arxiv">arXiv:{paper.id}</span>
        </div>
      </div>

      <section className="paper-detail-abstract">
        <h2>Abstract</h2>
        <MathMarkdown>{paper.abstract}</MathMarkdown>
      </section>

      {ext && (
        <section className="paper-detail-extraction">
          <div className="extraction-info">
            <span className="extraction-model">{ext.model}</span>
            <span className="extraction-tokens">{ext.total_tokens?.toLocaleString()} tokens</span>
            <CostBadge cost={ext.cost} />
          </div>
        </section>
      )}

      <section className="paper-detail-problems">
        <h2>
          Open Problems
          <span className="paper-detail-problem-count">{paper.problems.length}</span>
        </h2>
        {paper.problems.length > 0 ? (
          <div className="paper-detail-problem-list">
            {paper.problems.map(prob => (
              <div key={prob.id} className="paper-detail-problem-card">
                <h3>
                  <Link to={`/paper/${paper.id}/problem/${prob.id}`}>{prob.name}</Link>
                </h3>
                {prob.location && <div className="paper-detail-problem-loc">{prob.location}</div>}
                {prob.summary && <div className="paper-detail-problem-summary">{prob.summary}</div>}
              </div>
            ))}
          </div>
        ) : (
          <p className="paper-detail-no-problems">
            {ext ? 'No open problems were found in this paper.' : 'This paper has not been extracted yet.'}
          </p>
        )}
      </section>
    </div>
  )
}

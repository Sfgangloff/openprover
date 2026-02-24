import { useMemo, useTransition } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useData } from '../hooks/useData'
import ProblemSummary from '../components/ProblemSummary'
import CostBadge from '../components/CostBadge'
import './PaperList.css'

function formatDate(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function PaperCard({ paper }) {
  const hasProblems = paper.problems.length > 0
  const extracted = paper.extraction && !paper.extraction.error

  return (
    <article className="paper-card">
      <div className="paper-card-header">
        <h3 className="paper-card-title">
          <Link to={`/paper/${paper.id}`}>{paper.title}</Link>
        </h3>
        {extracted && paper.extraction.cost && (
          <CostBadge cost={paper.extraction.cost} />
        )}
      </div>

      <div className="paper-card-meta">
        <span className="paper-card-authors">{paper.authors.join(', ')}</span>
        <span className="paper-card-sep">&middot;</span>
        <span className="paper-card-date">{formatDate(paper.published)}</span>
        {paper.categories.map(cat => (
          <span key={cat} className="paper-card-tag">{cat}</span>
        ))}
        <a href={paper.pdf_url} target="_blank" rel="noopener" className="paper-card-pdf">PDF</a>
      </div>

      {hasProblems ? (
        <ul className="paper-card-problems">
          {paper.problems.map(prob => (
            <ProblemSummary key={prob.id} problem={prob} paperId={paper.id} />
          ))}
        </ul>
      ) : extracted ? (
        <div className="paper-card-status paper-card-status--none">No problems found</div>
      ) : paper.extraction?.error ? (
        <div className="paper-card-status paper-card-status--error">Extraction failed</div>
      ) : (
        <div className="paper-card-status paper-card-status--pending">Not yet extracted</div>
      )}
    </article>
  )
}

export default function PaperList() {
  const { papers, stats } = useData()
  const [searchParams, setSearchParams] = useSearchParams()
  const filterProblems = searchParams.get('filter') === 'problems'

  const filtered = useMemo(() => {
    if (!filterProblems) return papers
    return papers.filter(p => p.problems.length > 0)
  }, [papers, filterProblems])

  const [isPending, startTransition] = useTransition()

  const toggle = () => {
    startTransition(() => {
      if (filterProblems) {
        searchParams.delete('filter')
      } else {
        searchParams.set('filter', 'problems')
      }
      setSearchParams(searchParams)
    })
  }

  return (
    <div className="paper-list">
      <div className="paper-list-stats">
        {stats.total_papers} papers &middot; {stats.papers_with_problems} with problems &middot; {stats.total_problems} problems &middot; <span className="paper-list-cost">{stats.total_cost} total</span>
      </div>

      <div className="paper-list-toolbar">
        <div className="paper-list-count">
          Showing {filtered.length} paper{filtered.length !== 1 ? 's' : ''}
        </div>
        <button
          className={`paper-list-toggle${filterProblems ? ' paper-list-toggle--active' : ''}`}
          onClick={toggle}
          type="button"
        >
          <span className="toggle-track"><span className="toggle-knob" /></span>
          Only with problems
        </button>
      </div>

      <div className={`paper-list-cards${isPending ? ' paper-list-cards--loading' : ''}`}>
        {filtered.map(paper => (
          <PaperCard key={paper.id} paper={paper} />
        ))}
      </div>
    </div>
  )
}

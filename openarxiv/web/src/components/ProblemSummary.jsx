import { Link } from 'react-router-dom'

export default function ProblemSummary({ problem, paperId }) {
  return (
    <li className="problem-summary">
      <Link to={`/paper/${paperId}/problem/${problem.id}`} className="problem-summary-name">
        {problem.name}
      </Link>
      {problem.summary && (
        <span className="problem-summary-tldr"> &mdash; {problem.summary}</span>
      )}
    </li>
  )
}

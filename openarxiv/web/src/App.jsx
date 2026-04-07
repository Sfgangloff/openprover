import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { useData } from './hooks/useData'
import PaperList from './pages/PaperList'
import PaperDetail from './pages/PaperDetail'
import ProblemDetail from './pages/ProblemDetail'
import './App.css'

function NavBar() {
  const { stats } = useData()
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link to="/" className="navbar-title">ArXiv Open Problems</Link>
        {stats && (
          <div className="navbar-stats">
            <span>{stats.total_papers} papers</span>
            <span className="navbar-sep">/</span>
            <span>{stats.total_problems} problems</span>
            <span className="navbar-sep">/</span>
            <span className="navbar-cost">{stats.total_cost}</span>
          </div>
        )}
      </div>
    </nav>
  )
}

function Breadcrumbs() {
  const location = useLocation()
  const { papers } = useData()

  const parts = location.pathname.split('/').filter(Boolean)
  // Routes: /, /paper/:id, /paper/:id/problem/:pid
  const crumbs = [{ label: 'Papers', to: '/' }]

  if (parts[0] === 'paper' && parts[1]) {
    const paper = papers.find(p => p.id === parts[1])
    const title = paper ? (paper.title.length > 50 ? paper.title.slice(0, 50) + '...' : paper.title) : parts[1]
    crumbs.push({ label: title, to: `/paper/${parts[1]}` })

    if (parts[2] === 'problem' && parts[3]) {
      const problem = paper?.problems.find(p => p.id === parts[3])
      const name = problem ? (problem.name.length > 40 ? problem.name.slice(0, 40) + '...' : problem.name) : parts[3]
      crumbs.push({ label: name, to: `/paper/${parts[1]}/problem/${parts[3]}` })
    }
  }

  if (crumbs.length <= 1) return null

  return (
    <div className="breadcrumbs">
      {crumbs.map((c, i) => (
        <span key={c.to}>
          {i > 0 && <span className="breadcrumb-sep">&rsaquo;</span>}
          {i < crumbs.length - 1 ? (
            <Link to={c.to} className="breadcrumb-link">{c.label}</Link>
          ) : (
            <span className="breadcrumb-current">{c.label}</span>
          )}
        </span>
      ))}
    </div>
  )
}

export default function App() {
  const { loading, error } = useData()

  if (loading) return <div className="app-loading">Loading...</div>
  if (error) return <div className="app-error">Failed to load data: {error}</div>

  return (
    <>
      <NavBar />
      <main className="main">
        <Breadcrumbs />
        <Routes>
          <Route path="/" element={<PaperList />} />
          <Route path="/paper/:paperId" element={<PaperDetail />} />
          <Route path="/paper/:paperId/problem/:problemId" element={<ProblemDetail />} />
        </Routes>
      </main>
    </>
  )
}

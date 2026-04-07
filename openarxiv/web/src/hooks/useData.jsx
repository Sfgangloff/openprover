import { createContext, useContext, useState, useEffect, useMemo } from 'react'

const REMOTE_URL = 'https://ipwzcwxgemjervyvptjb.supabase.co/storage/v1/object/public/openarxiv/data.json'
const LOCAL_URL = '/data.json'
const DATA_URL = import.meta.env.VITE_DATA_SOURCE === 'remote' ? REMOTE_URL : LOCAL_URL

const DataContext = createContext(null)

export function DataProvider({ children }) {
  const [papers, setPapers] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(DATA_URL)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(data => {
        setPapers(data.papers || [])
        setStats(data.stats || null)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  const value = useMemo(() => ({ papers, stats, loading, error }), [papers, stats, loading, error])

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>
}

export function useData() {
  const ctx = useContext(DataContext)
  if (!ctx) throw new Error('useData must be used within DataProvider')
  return ctx
}

export function usePaper(paperId) {
  const { papers } = useData()
  return useMemo(() => papers.find(p => p.id === paperId) || null, [papers, paperId])
}

export function useProblem(paperId, problemId) {
  const paper = usePaper(paperId)
  return useMemo(() => {
    if (!paper) return { paper: null, problem: null }
    const problem = paper.problems.find(p => p.id === problemId) || null
    return { paper, problem }
  }, [paper, problemId])
}

export function useAllProblems() {
  const { papers } = useData()
  return useMemo(() => {
    const list = []
    for (const paper of papers) {
      for (const problem of paper.problems) {
        list.push({ paper, problem })
      }
    }
    return list
  }, [papers])
}

export function useAdjacentProblems(paperId, problemId) {
  const all = useAllProblems()
  return useMemo(() => {
    const idx = all.findIndex(e => e.paper.id === paperId && e.problem.id === problemId)
    return {
      prev: idx > 0 ? all[idx - 1] : null,
      next: idx >= 0 && idx < all.length - 1 ? all[idx + 1] : null,
      index: idx,
      total: all.length,
    }
  }, [all, paperId, problemId])
}

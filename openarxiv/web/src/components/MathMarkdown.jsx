import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'

/**
 * remark-math converts $...$ to <code class="language-math math-inline">
 * and $$...$$ to <pre><code class="language-math math-display">.
 * We intercept these and render them as raw delimited text so
 * MathJax (loaded via CDN in index.html) can typeset them.
 */
const mathComponents = {
  code({ className, children, ...props }) {
    if (className === 'language-math math-inline') {
      return <span className="math-inline">{`$${children}$`}</span>
    }
    if (className === 'language-math math-display') {
      return <div className="math-display">{`$$${children}$$`}</div>
    }
    return <code className={className} {...props}>{children}</code>
  },
  // Block math gets wrapped in <pre> by default — unwrap it
  pre({ children }) {
    // If the child is our math-display div, just return it directly
    if (children?.props?.className === 'language-math math-display') {
      return mathComponents.code(children.props)
    }
    return <pre>{children}</pre>
  },
}

/**
 * Walk text nodes in `el` and wrap [Tag] patterns that match known
 * reference tags with <a href="#ref-Tag"> links.
 */
function linkifyRefs(el, references) {
  if (!references || references.length === 0) return

  const tagSet = new Set(references.map(r => r.tag))
  // Sort tags longest-first so greedy matching prefers longer tags
  const sortedTags = [...tagSet].sort((a, b) => b.length - a.length)

  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT)
  const nodes = []
  while (walker.nextNode()) {
    const node = walker.currentNode
    // Skip inside MathJax output, existing links, or math spans
    if (node.parentElement.closest('mjx-container, a, .math-inline, .math-display')) continue
    if (/\[/.test(node.textContent)) nodes.push(node)
  }

  for (const textNode of nodes) {
    const text = textNode.textContent
    const frag = document.createDocumentFragment()
    let lastIndex = 0
    const re = /\[([^\]]+)\]/g
    let match

    while ((match = re.exec(text)) !== null) {
      const bracketContent = match[1]
      // Find the first known tag that appears in this bracket group
      const hitTag = sortedTags.find(tag => bracketContent.includes(tag))
      if (!hitTag) continue

      // Text before the bracket
      if (match.index > lastIndex) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)))
      }
      const link = document.createElement('a')
      link.href = `#ref-${hitTag}`
      link.className = 'ref-link'
      link.textContent = match[0]
      link.addEventListener('click', (e) => {
        e.preventDefault()
        const target = document.getElementById(`ref-${hitTag}`)
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'center' })
          target.classList.add('ref-highlight')
          setTimeout(() => target.classList.remove('ref-highlight'), 1500)
        }
      })
      frag.appendChild(link)
      lastIndex = match.index + match[0].length
    }

    if (lastIndex > 0) {
      if (lastIndex < text.length) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex)))
      }
      textNode.parentNode.replaceChild(frag, textNode)
    }
  }
}

export default function MathMarkdown({ children, className, references }) {
  const containerRef = useRef(null)
  const [hasError, setHasError] = useState(false)

  const typeset = useCallback(() => {
    const el = containerRef.current
    if (!el || !window.MathJax?.typesetPromise) return

    window.MathJax.typesetClear?.([el])

    window.MathJax.typesetPromise([el])
      .then(() => {
        const errors = el.querySelectorAll('mjx-merror')
        setHasError(errors && errors.length > 0)
        linkifyRefs(el, references)
      })
      .catch(() => {
        setHasError(true)
      })
  }, [references])

  useEffect(() => {
    if (!children) return

    // MathJax may not be loaded yet (async CDN script)
    if (window.MathJax?.typesetPromise) {
      typeset()
    } else {
      const id = setInterval(() => {
        if (window.MathJax?.typesetPromise) {
          clearInterval(id)
          typeset()
        }
      }, 100)
      return () => clearInterval(id)
    }
  }, [children, typeset])

  if (!children) return null

  return (
    <div ref={containerRef} className={`math-markdown ${className || ''}`}>
      <ReactMarkdown
        remarkPlugins={[remarkMath]}
        components={mathComponents}
      >
        {children}
      </ReactMarkdown>
      {hasError && (
        <details className="math-error">
          <summary>Some math failed to render. Original source:</summary>
          <pre className="math-source">{children}</pre>
        </details>
      )}
    </div>
  )
}

export default function CostBadge({ cost }) {
  if (!cost) return null
  return <span className="cost-badge">{cost}</span>
}

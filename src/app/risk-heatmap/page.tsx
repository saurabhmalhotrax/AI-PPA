import RiskNetworkHeatmap from '@/app/components/RiskNetworkHeatmap'

export default function Page() {
  const nodes = [
    { id: 'v1', type: 'vendor', risk: 80 },
    { id: 'i1', type: 'invoice', risk: 45 },
    { id: 'c1', type: 'contract', risk: 60 },
    // …
  ]
  const links = [
    { source: 'v1', target: 'i1' },
    { source: 'i1', target: 'c1' },
    // …
  ]

  return <RiskNetworkHeatmap nodes={nodes} links={links} />
} 
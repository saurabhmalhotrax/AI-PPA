'use client'
import React, { useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { OrthographicView, COORDINATE_SYSTEM } from '@deck.gl/core'
import { HeatmapLayer } from '@deck.gl/aggregation-layers'
import { ScatterplotLayer, ArcLayer } from '@deck.gl/layers'
import { scaleLinear } from 'd3-scale'

type Node = {
  id: string
  type: 'vendor' | 'invoice' | 'contract'
  risk: number // 0–100
}
type Link = { source: string; target: string }

interface Props {
  nodes: Node[]
  links: Link[]
  width?: number
  height?: number
}

export default function RiskNetworkHeatmap({
  nodes,
  links,
  width = 800,
  height = 600,
}: Props) {
  // map types to vertical bands
  const typeOrder = ['vendor', 'invoice', 'contract']
  const yBand = height / typeOrder.length

  // assign each node a fixed x/y based on risk & type
  const data = useMemo(
    () =>
      nodes.map((n) => ({
        ...n,
        position: [
          (n.risk / 100) * width,
          (typeOrder.indexOf(n.type) + 0.5) * yBand,
        ] as [number, number],
      })),
    [nodes, width, height]
  )

  const colorScale = scaleLinear<string>()
    .domain([0, 100])
    .range(['#2ecc71', '#e74c3c'])

  const layers = [
    new HeatmapLayer<Node>({
      id: 'heat',
      data,
      getPosition: (d) => d.position,
      getWeight: (d) => d.risk,
      radiusPixels: 60,
      intensity: 1,
    }),
    new ArcLayer<Link>({
      id: 'links',
      data: links,
      getSourcePosition: (d) =>
        data.find((n) => n.id === d.source)!.position,
      getTargetPosition: (d) =>
        data.find((n) => n.id === d.target)!.position,
      getSourceColor: [100, 100, 255],
      getTargetColor: [255, 100, 100],
      strokeWidth: 2,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    }),
    new ScatterplotLayer<Node>({
      id: 'nodes',
      data,
      getPosition: (d) => d.position,
      getFillColor: (d) => colorScale(d.risk).match(/\d+/g)!.map(Number),
      getRadius: 8,
      pickable: true,
      coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
    }),
  ]

  return (
    <DeckGL
      views={[new OrthographicView()]}
      initialViewState={{
        target: [width / 2, height / 2, 0],
        zoom: 0,
      }}
      controller
      layers={layers}
      width={width}
      height={height}
    />
  )
} 
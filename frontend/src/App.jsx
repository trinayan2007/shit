import React, { useCallback, useEffect, useState } from 'react'
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts'
import { api } from './api.js'

const RISK_COLORS = { LOW: '#3fb950', MEDIUM: '#d29922', HIGH: '#f85149' }

function fmt(n) {
  if (n === null || n === undefined) return '-'
  return typeof n === 'number' ? n.toLocaleString('en-US') : String(n)
}

function shortTime(iso) {
  if (!iso) return '-'
  return iso.replace('T', ' ').replace('Z', '')
}

/* ---------------- header ---------------- */
function Header({ health, onRefresh }) {
  const model = health?.model_status || 'UNKNOWN'
  return (
    <header className="hdr">
      <div>
        <h1>INSIDER THREAT DETECTION</h1>
        <div className="sub">
          Hybrid GRU-Autoencoder · CloudTrail · MITRE ATT&amp;CK · Automated Response
        </div>
      </div>
      <div className="status-group">
        <span className="badge ok">SYSTEM <b>{health?.status === 'online' ? 'ONLINE' : 'OFFLINE'}</b></span>
        <span className="badge ok">BACKEND <b>{health?.backend_status || '-'}</b></span>
        <span className={`badge ${model === 'LOADED' ? 'on' : 'bad'}`}>MODEL <b>{model}</b></span>
        <span className="badge warn">RESPONSE <b>{health?.mode?.includes('SIMULATED') ? 'SIMULATED' : 'REAL'}</b></span>
        <button className="btn" onClick={onRefresh}>↻ Refresh</button>
      </div>
    </header>
  )
}

/* ---------------- KPI cards ---------------- */
function Kpis({ dash }) {
  const k = dash?.kpis || {}
  return (
    <div className="kpis">
      <div className="kpi accent">
        <div className="label">Total Events</div>
        <div className="value">{fmt(k.total_events)}</div>
        <div className="hint">deduplicated CloudTrail events processed</div>
      </div>
      <div className="kpi">
        <div className="label">Unique Users / Principals</div>
        <div className="value">{fmt(k.unique_principals)}</div>
        <div className="hint">resolved identity keys</div>
      </div>
      <div className="kpi warn">
        <div className="label">Detected Anomalies</div>
        <div className="value">{fmt(k.detected_anomalies)}</div>
        <div className="hint">windows ≥ calibrated threshold</div>
      </div>
      <div className="kpi danger">
        <div className="label">High-Risk Incidents</div>
        <div className="value">{fmt(k.high_risk_incidents)}</div>
        <div className="hint">eligible for mitigation</div>
      </div>
    </div>
  )
}

/* ---------------- charts ---------------- */
function Timeline({ data }) {
  const rows = (data || []).map((d) => ({ ...d, period: d.period.slice(2) }))
  return (
    <div className="panel">
      <h2>Activity / Threat Timeline (detections per month)</h2>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={rows} margin={{ top: 5, right: 8, left: -18, bottom: 0 }}>
          <defs>
            <linearGradient id="cAll" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#2f81f7" stopOpacity={0.5} />
              <stop offset="95%" stopColor="#2f81f7" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="cHigh" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#f85149" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#f85149" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1f2a38" strokeDasharray="3 3" />
          <XAxis dataKey="period" tick={{ fill: '#7d8fa5', fontSize: 10 }} />
          <YAxis tick={{ fill: '#7d8fa5', fontSize: 10 }} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: '#121821', border: '1px solid #1f2a38', fontSize: 12 }}
            labelStyle={{ color: '#d7e0ea' }}
          />
          <Area type="monotone" dataKey="count" name="All detections" stroke="#2f81f7" fill="url(#cAll)" strokeWidth={1.6} />
          <Area type="monotone" dataKey="high" name="High risk" stroke="#f85149" fill="url(#cHigh)" strokeWidth={1.6} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function RiskPie({ dist }) {
  const rows = Object.entries(dist || {}).map(([name, value]) => ({ name, value }))
  return (
    <div className="panel">
      <h2>Threat Distribution</h2>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie data={rows} dataKey="value" nameKey="name"
               innerRadius={52} outerRadius={82} paddingAngle={2} isAnimationActive={false}>
            {rows.map((r) => <Cell key={r.name} fill={RISK_COLORS[r.name]} />)}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#121821', border: '1px solid #1f2a38', fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

function MitreBar({ data }) {
  const rows = (data || []).map((d) => ({
    name: d.technique_id === 'none' ? 'none' : d.technique_id,
    count: d.count,
  }))
  return (
    <div className="panel">
      <h2>MITRE ATT&amp;CK Techniques (detected threats)</h2>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 12, left: 18, bottom: 0 }}>
          <CartesianGrid stroke="#1f2a38" strokeDasharray="3 3" />
          <XAxis type="number" allowDecimals={false} tick={{ fill: '#7d8fa5', fontSize: 10 }} />
          <YAxis type="category" dataKey="name" width={78} tick={{ fill: '#7d8fa5', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#121821', border: '1px solid #1f2a38', fontSize: 12 }}
          />
          <Bar dataKey="count" fill="#2f81f7" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ---------------- recent threats ---------------- */
function RecentThreats({ items, onOpen }) {
  return (
    <div className="panel">
      <h2>Recent Threats (highest anomaly score)</h2>
      {(!items || items.length === 0) && <div className="empty">No non-LOW detections</div>}
      {items && items.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>User</th><th>Event</th><th>Score</th><th>Risk</th><th>MITRE</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {items.map((t) => (
              <tr key={t.id} onClick={() => onOpen(t.id)}>
                <td>{t.display_name}</td>
                <td className="muted">{(t.top_events || []).join(', ')}</td>
                <td>{t.anomaly_score.toFixed(4)}</td>
                <td><span className={`pill ${t.risk_level}`}>{t.risk_level}</span></td>
                <td className="tech">{t.mitre_technique_id || '—'}</td>
                <td>
                  <span className={`pill ${t.response_status === 'EXECUTED' ? 'blocked' : 'status'}`}>
                    {t.response_status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

/* ---------------- activity table ---------------- */
function ActivityTable({ state, setState, onOpen }) {
  const { items, total, limit, offset, risk, q } = state
  const page = Math.floor(offset / limit) + 1
  const pages = Math.max(1, Math.ceil(total / limit))

  const set = (patch) => setState((s) => ({ ...s, ...patch, offset: patch.offset ?? 0 }))

  return (
    <div className="panel">
      <h2>Detected Activity ({fmt(total)} windows)</h2>
      <div className="toolbar">
        <select value={risk} onChange={(e) => set({ risk: e.target.value })}>
          <option value="">All risk levels</option>
          <option value="HIGH">HIGH</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="LOW">LOW</option>
        </select>
        <input
          placeholder="Search user / event…"
          value={q}
          onChange={(e) => set({ q: e.target.value })}
          onKeyDown={(e) => e.key === 'Enter' && setState((s) => ({ ...s, search: s.q }))}
          style={{ width: 220 }}
        />
        <button className="btn" onClick={() => setState((s) => ({ ...s, search: s.q }))}>
          Search
        </button>
        <div className="spacer" />
        <span className="muted">click a row for details</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th><th>User</th><th>Event</th><th>Service</th><th>Source IP</th>
            <th>Score</th><th>Threat</th><th>MITRE</th><th>Response</th>
          </tr>
        </thead>
        <tbody>
          {items.map((e) => {
            const peak = e.peak_event || {}
            const ev = (e.top_events || []).join(', ')
            return (
              <tr key={e.id} onClick={() => onOpen(e.id)}>
                <td>{shortTime(e.window_start)}</td>
                <td>{e.principal?.display_name}</td>
                <td className="muted">{ev}</td>
                <td className="muted">{peak.eventSource || '—'}</td>
                <td className="muted">{peak.sourceIPAddress || '—'}</td>
                <td>{e.anomaly_score.toFixed(4)}</td>
                <td><span className={`pill ${e.risk_level}`}>{e.risk_level}</span></td>
                <td className="tech">{e.mitre_technique_id || '—'}</td>
                <td>
                  <span className={`pill ${e.response_status === 'EXECUTED' ? 'blocked' : 'status'}`}>
                    {e.response_status}
                  </span>
                </td>
              </tr>
            )
          })}
          {items.length === 0 && (
            <tr><td colSpan={9} className="empty">No rows match the current filter</td></tr>
          )}
        </tbody>
      </table>
      <div className="pager">
        <button className="btn" disabled={offset === 0}
          onClick={() => setState((s) => ({ ...s, offset: Math.max(0, s.offset - s.limit) }))}>
          ← Prev
        </button>
        <span>page {page} / {pages}</span>
        <button className="btn" disabled={offset + limit >= total}
          onClick={() => setState((s) => ({ ...s, offset: s.offset + s.limit }))}>
          Next →
        </button>
      </div>
    </div>
  )
}

/* ---------------- threat details drawer ---------------- */
function Drawer({ threatId, onClose, onChanged }) {
  const [detail, setDetail] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    if (!threatId) return
    setError('')
    api.threat(threatId).then(setDetail).catch((e) => setError(e.message))
  }, [threatId])

  useEffect(load, [load])

  const block = async () => {
    setBusy(true)
    setError('')
    try {
      const res = await api.block(threatId)
      setDetail((d) => ({ ...d, blocked: res, response_status: res.status }))
      if (onChanged) onChanged()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (!threatId) return null
  const peak = detail?.peak_event || {}
  const mitre = detail?.mitre || {}
  const ev = detail?.evidence_summary || {}

  return (
    <>
      <div className="overlay" onClick={onClose} />
      <aside className="drawer">
        <h3>
          Threat #{threatId}
          <button className="close" onClick={onClose}>×</button>
        </h3>
        {!detail && !error && <div className="empty">Loading…</div>}
        {error && <div className="alert-error">{error}</div>}

        {detail && (
          <>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span className={`pill ${detail.risk_level}`}>{detail.risk_level}</span>
              <span className={`pill ${detail.response_status === 'EXECUTED' ? 'blocked' : 'status'}`}>
                {detail.response_status}
              </span>
              <span className="pill status">{detail.split}</span>
            </div>

            <dl className="dl">
              <dt>Threat ID</dt><dd>{detail.id}</dd>
              <dt>User / Principal</dt>
              <dd>{detail.principal?.display_name} ({detail.principal?.principal_key})</dd>
              <dt>Principal status</dt>
              <dd>{detail.principal?.status}</dd>
              <dt>Window start</dt><dd>{detail.window_start}</dd>
              <dt>Window end</dt><dd>{detail.window_end}</dd>
              <dt>AWS Event (peak)</dt><dd>{peak.eventName || '—'}</dd>
              <dt>AWS Service</dt><dd>{peak.eventSource || '—'}</dd>
              <dt>Source IP</dt><dd>{peak.sourceIPAddress || '—'}</dd>
              <dt>Region</dt><dd>{peak.awsRegion || '—'}</dd>
              <dt>Error code</dt><dd>{peak.errorCode || '—'}</dd>
              <dt>Peak step error</dt>
              <dd>{peak.peak_step_error != null ? peak.peak_step_error.toFixed(6) : '—'}</dd>
              <dt>Anomaly score</dt><dd>{detail.anomaly_score.toFixed(6)}</dd>
              <dt>Threshold</dt><dd>{detail.threshold.toFixed(6)} (calibrated p99 of train errors)</dd>
            </dl>

            <div className="section">
              <h4>Detection evidence / reasons</h4>
              {(detail.reasons || []).map((r, i) => (
                <div className="reason" key={i}>{r}</div>
              ))}
              <div className="evidence-tags" style={{ marginTop: 8 }}>
                {(ev.sensitive_events || []).map((s) => (
                  <span className="tag" key={s} style={{ borderColor: '#d29922', color: '#d29922' }}>{s}</span>
                ))}
                {(ev.distinct_event_names || []).slice(0, 8).map((s) => (
                  <span className="tag" key={s}>{s}</span>
                ))}
                {ev.auth_errors > 0 && (
                  <span className="tag" style={{ borderColor: '#f85149', color: '#f85149' }}>
                    auth errors ×{ev.auth_errors}
                  </span>
                )}
              </div>
            </div>

            <div className="section">
              <h4>MITRE ATT&amp;CK mapping</h4>
              {mitre.status === 'mapped' ? (
                <>
                  <div className="tech" style={{ fontSize: 13 }}>
                    {mitre.technique_id} — {mitre.technique_name}
                  </div>
                  <div className="muted" style={{ margin: '6px 0', fontSize: 11.5 }}>
                    Tactic: {mitre.tactic}
                  </div>
                  <div className="reason" style={{ borderLeftColor: '#2f81f7' }}>
                    {mitre.rationale}
                  </div>
                  <div className="evidence-tags">
                    {(mitre.supporting_behavior || []).map((s) => (
                      <span className="tag" key={s}>{s}</span>
                    ))}
                  </div>
                </>
              ) : (
                <div className="muted" style={{ fontSize: 11.5 }}>
                  {mitre.rationale || 'No confident MITRE mapping'}
                </div>
              )}
            </div>

            <div className="section">
              <h4>Automated response</h4>
              <div className="muted" style={{ fontSize: 11.5, marginBottom: 6 }}>
                Policy: LOW → log · MEDIUM → escalate · HIGH → mitigation (operator-triggered)
              </div>
              {detail.risk_level === 'HIGH' ? (
                <div className="mitigation">
                  {detail.response_status === 'EXECUTED' ? (
                    <div className="exec-box">
                      ✓ Response Executed<br />
                      Status: BLOCKED<br />
                      SIMULATED RESPONSE
                    </div>
                  ) : detail.blocked ? (
                    <div className="exec-box">
                      ✓ Response Executed ({detail.blocked.action})<br />
                      Status: {detail.blocked.principal_status}<br />
                      {detail.blocked.label}
                    </div>
                  ) : (
                    <button className="block-btn" onClick={block} disabled={busy}>
                      {busy ? 'EXECUTING…' : 'BLOCK / MITIGATE'}
                    </button>
                  )}
                  <div className="sim-note">
                    SIMULATED RESPONSE — updates local state only; no real AWS account is disabled.
                  </div>
                </div>
              ) : (
                <div className="muted" style={{ fontSize: 11.5 }}>
                  Blocking is only available for HIGH risk threats. Current action:{' '}
                  <b>{detail.response_action}</b> ({detail.response_status})
                </div>
              )}
              {detail.responses && detail.responses.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  {detail.responses.map((r) => (
                    <div className="exec-box" key={r.id}>
                      {r.action} · {r.mode} · {r.status} · {r.executed_at}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </>
        )}
      </aside>
    </>
  )
}

/* ---------------- app ---------------- */
export default function App() {
  const [health, setHealth] = useState(null)
  const [dash, setDash] = useState(null)
  const [table, setTable] = useState({
    items: [], total: 0, limit: 25, offset: 0, risk: '', q: '', search: '',
  })
  const [threatId, setThreatId] = useState(null)
  const [loadError, setLoadError] = useState('')

  const loadAll = useCallback(async () => {
    try {
      const [h, d] = await Promise.all([api.health(), api.dashboard()])
      setHealth(h)
      setDash(d)
      setLoadError('')
    } catch (e) {
      setLoadError(`Backend unreachable: ${e.message} — start it with: cd backend && uvicorn app.main:app --port 8001`)
    }
  }, [])

  const loadTable = useCallback(async () => {
    try {
      const res = await api.events({
        limit: table.limit,
        offset: table.offset,
        risk: table.risk,
        q: table.search,
      })
      setTable((s) => ({ ...s, items: res.items, total: res.total }))
    } catch (e) {
      setLoadError(e.message)
    }
  }, [table.limit, table.offset, table.risk, table.search])

  useEffect(() => { loadAll() }, [loadAll])
  useEffect(() => { loadTable() }, [loadTable])
  // light polling keeps the console alive during a demo
  useEffect(() => {
    const t = setInterval(loadAll, 30000)
    return () => clearInterval(t)
  }, [loadAll])

  return (
    <div className="app">
      <Header health={health} onRefresh={() => { loadAll(); loadTable() }} />
      {loadError && <div className="alert-error" style={{ marginBottom: 12 }}>{loadError}</div>}
      <Kpis dash={dash} />
      <div className="grid-3">
        <Timeline data={dash?.timeline} />
        <RiskPie dist={dash?.risk_distribution} />
        <MitreBar data={dash?.mitre_distribution} />
      </div>
      <div style={{ marginBottom: 14 }}>
        <RecentThreats items={dash?.recent_threats} onOpen={setThreatId} />
      </div>
      <ActivityTable state={table} setState={setTable} onOpen={setThreatId} />

      <div className="footer-note">
        Anomaly scores are GRU-Autoencoder reconstruction errors (masked MSE) ·
        threshold calibrated from the training error distribution ·
        MITRE mapping is a separate evidence-based layer ·
        blocking is SIMULATED by default
      </div>

      <Drawer threatId={threatId} onClose={() => setThreatId(null)} onChanged={() => { loadAll(); loadTable() }} />
    </div>
  )
}

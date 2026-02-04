import { useMemo, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:5000'

function prettyNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  return n.toFixed(3)
}

function App() {
  const [file, setFile] = useState(null)
  const [url, setUrl] = useState('')

  const [loadingUpload, setLoadingUpload] = useState(false)
  const [loadingForecast, setLoadingForecast] = useState(false)

  const [statusUpload, setStatusUpload] = useState('')
  const [statusForecast, setStatusForecast] = useState('')
  const [error, setError] = useState('')

  const [eda, setEda] = useState(null)
  const [forecast, setForecast] = useState(null)

  const chartData = useMemo(() => {
    if (!forecast?.timestamps || !forecast?.predictions) return []
    const targets = Object.keys(forecast.predictions)
    const primary = targets[0]
    const preds = forecast.predictions[primary] || []
    const acts = forecast.actuals?.[primary] || []
    return forecast.timestamps.map((ts, idx) => ({
      ts,
      actual: acts[idx] ?? null,
      predicted: preds[idx] ?? null,
    }))
  }, [forecast])

  async function handleUploadSource(type) {
    setError('')
    setStatusForecast('')
    setForecast(null)
    setEda(null)
    setLoadingUpload(true)
    setStatusUpload(type === 'file' ? 'Uploading CSV and running EDA…' : 'Fetching from URL and running EDA…')
    try {
      let res
      if (type === 'file') {
        if (!file) {
          setError('Please choose a CSV file first.')
          return
        }
        const formData = new FormData()
        formData.append('file', file)
        res = await fetch(`${API_BASE}/upload`, {
          method: 'POST',
          body: formData,
        })
      } else {
        if (!url.trim()) {
          setError('Please provide a URL pointing to a CSV file.')
          return
        }
        res = await fetch(`${API_BASE}/fetch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url }),
        })
      }

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || `Backend responded with ${res.status}`)
      }

      const data = await res.json()
      setEda(data)
      setStatusUpload('EDA completed. Processed data is ready for forecasting.')
    } catch (e) {
      setError(e.message || 'Failed to upload/fetch data.')
      setStatusUpload('')
    } finally {
      setLoadingUpload(false)
    }
  }

  async function handleForecast() {
    setError('')
    setLoadingForecast(true)
    setStatusForecast('Training XGBoost and generating forecast…')
    try {
      const res = await fetch(`${API_BASE}/forecast`, {
        method: 'POST',
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.error || `Backend responded with ${res.status}`)
      }
      const data = await res.json()
      setForecast(data)
      setStatusForecast('Forecast ready.')
    } catch (e) {
      setError(e.message || 'Failed to run forecast.')
      setStatusForecast('')
    } finally {
      setLoadingForecast(false)
    }
  }

  return (
    <div className="app-root">
      <div className="app-shell">
        <header className="app-header">
          <h1 className="app-title">ForecastX – Time Series Lab</h1>
          <p className="app-subtitle">
            React + Flask + XGBoost system for time series EDA, feature engineering, and forecasting.
          </p>
        </header>

        <div className="grid">
          {/* Ingestion */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">1 · Data Ingestion</h2>
              <span className="pill">Upload or URL</span>
            </div>
            <div className="panel-body">
              <div>
                <div className="field-label">Upload CSV</div>
                <p className="field-description">Any time-indexed CSV – the API auto-detects the datetime column.</p>
                <div className="file-input">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => {
                      setFile(e.target.files?.[0] || null)
                    }}
                  />
                  <button
                    className="btn btn-secondary"
                    onClick={() => handleUploadSource('file')}
                    disabled={loadingUpload}
                  >
                    {loadingUpload ? 'Processing…' : 'Run EDA'}
                  </button>
                </div>
              </div>

              <div>
                <div className="field-label">Or Fetch from URL</div>
                <p className="field-description">Paste a CSV URL from a public endpoint or raw GitHub link.</p>
                <input
                  className="text-input"
                  type="url"
                  placeholder="https://…/timeseries.csv"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                />
              </div>

              <div className="button-row">
                <button
                  className="btn btn-secondary"
                  onClick={() => handleUploadSource('url')}
                  disabled={loadingUpload}
                >
                  {loadingUpload ? 'Fetching…' : 'Fetch & Run EDA'}
                </button>
              </div>

              {statusUpload && <div className="status-text success">{statusUpload}</div>}
            </div>
          </section>

          {/* EDA */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">2 · EDA & Features</h2>
              <span className="pill">Pandas · NumPy · Seaborn</span>
            </div>
            <div className="panel-body eda-section">
              {eda ? (
                <>
                  <div className="eda-grid">
                    <div className="eda-card">
                      <div className="eda-card-title">Summary Statistics</div>
                      <div className="eda-card-body">
                        <ul className="eda-list">
                          {Object.entries(eda.description || {}).slice(0, 4).map(([metric, cols]) => {
                            const firstCol = cols && typeof cols === 'object' ? Object.keys(cols)[0] : null
                            const firstVal = firstCol ? cols[firstCol] : null
                            return (
                              <li key={metric}>
                                <strong>{metric}:</strong> {prettyNumber(firstVal)}
                              </li>
                            )
                          })}
                        </ul>
                      </div>
                    </div>

                    <div className="eda-card">
                      <div className="eda-card-title">Stationarity (ADF)</div>
                      <div className="eda-card-body">
                        <ul className="eda-list">
                          <li>
                            <strong>ADF Statistic:</strong>{' '}
                            {prettyNumber(eda.stationarity?.['ADF Statistic'])}
                          </li>
                          <li>
                            <strong>p-value:</strong> {prettyNumber(eda.stationarity?.['p-value'])}
                          </li>
                          <li>
                            <span className="eda-tag">
                              <span className="eda-tag-dot" />
                              {String(eda.stationarity?.Stationary)}
                            </span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div className="eda-card">
                    <div className="eda-card-title">Resampled Views</div>
                    <div className="eda-card-body">
                      <ul className="eda-list">
                        <li>
                          <strong>Daily:</strong>{' '}
                          {eda.resample?.daily ? Object.keys(eda.resample.daily).length : 0} points
                        </li>
                        <li>
                          <strong>Weekly:</strong>{' '}
                          {eda.resample?.weekly ? Object.keys(eda.resample.weekly).length : 0} points
                        </li>
                        <li>
                          <strong>Monthly:</strong>{' '}
                          {eda.resample?.monthly ? Object.keys(eda.resample.monthly).length : 0} points
                        </li>
                      </ul>
                    </div>
                  </div>

                  <div className="eda-card">
                    <div className="eda-card-title">Outliers</div>
                    <div className="eda-card-body">
                      <span>
                        {Array.isArray(eda.outliers?.outliers)
                          ? `${eda.outliers.outliers.length} detected using Z-score > 3`
                          : 'No numeric series detected.'}
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <p className="field-description">
                  Run EDA first to see summary statistics, stationarity, resampling, and outlier counts here.
                </p>
              )}
            </div>
          </section>

          {/* Forecast */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">3 · XGBoost Forecast</h2>
              <span className="pill">Model & Visuals</span>
            </div>
            <div className="panel-body">
              <div>
                <div className="field-label">Run Forecast</div>
                <p className="field-description">
                  Uses lag features, calendar features, and an XGBoost regressor per numeric target.
                </p>
              </div>

              <div className="button-row">
                <button
                  className="btn btn-primary"
                  onClick={handleForecast}
                  disabled={loadingForecast || !eda}
                >
                  {loadingForecast ? 'Training…' : 'Train & Forecast'}
                </button>
              </div>

              {statusForecast && <div className="status-text success">{statusForecast}</div>}

              {error && <div className="status-text error">{error}</div>}

              {forecast && (
                <>
                  <div className="eda-card">
                    <div className="eda-card-title">Evaluation Metrics</div>
                    <div className="eda-card-body">
                      <table className="metrics-table">
                        <thead>
                          <tr>
                            <th>Target</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>MAPE</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(forecast.metrics || {}).map(([target, m]) => (
                            <tr key={target}>
                              <td>{target}</td>
                              <td>{prettyNumber(m.MAE)}</td>
                              <td>{prettyNumber(m.RMSE)}</td>
                              <td>{prettyNumber((m.MAPE || 0) * 100)}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {chartData.length > 0 && (
                    <div className="chart-container">
                      <div className="chart-inner">
                        <div className="chart-legend">
                          <span>
                            <span className="legend-dot actual" /> Actual
                          </span>
                          <span>
                            <span className="legend-dot predicted" /> XGBoost
                          </span>
                        </div>
                        <svg width="100%" height="180">
                          {(() => {
                            const width = 600
                            const height = 140
                            const padding = 20
                            const xs = chartData.map((_, i) => i)
                            const ys = chartData
                              .flatMap((d) => [d.actual, d.predicted])
                              .filter((v) => v !== null && v !== undefined)
                            if (!ys.length) return null
                            const xScale = (i) =>
                              padding + (i / Math.max(xs.length - 1, 1)) * (width - padding * 2)
                            const minY = Math.min(...ys)
                            const maxY = Math.max(...ys)
                            const yScale = (v) =>
                              height - padding - ((v - minY) / Math.max(maxY - minY, 1e-6)) * (height - padding * 2)

                            const linePath = (valuesKey) => {
                              let path = ''
                              chartData.forEach((d, i) => {
                                const v = d[valuesKey]
                                if (v === null || v === undefined) return
                                const x = xScale(i)
                                const y = yScale(v)
                                path += path ? ` L ${x},${y}` : `M ${x},${y}`
                              })
                              return path
                            }

                            const actualPath = linePath('actual')
                            const predictedPath = linePath('predicted')

                            return (
                              <g transform="translate(0,10)">
                                <rect
                                  x={padding}
                                  y={padding}
                                  width={width - padding * 2}
                                  height={height - padding * 2}
                                  fill="url(#bg)"
                                  stroke="rgba(15,23,42,0.9)"
                                  rx="10"
                                />
                                <defs>
                                  <linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">
                                    <stop offset="0%" stopColor="rgba(15,23,42,1)" />
                                    <stop offset="100%" stopColor="rgba(2,6,23,1)" />
                                  </linearGradient>
                                </defs>
                                {Array.from({ length: 5 }).map((_, i) => {
                                  const y =
                                    padding +
                                    (i / 4) * (height - padding * 2)
                                  return (
                                    <line
                                      key={i}
                                      x1={padding}
                                      x2={width - padding}
                                      y1={y}
                                      y2={y}
                                      stroke="rgba(31,41,55,0.9)"
                                      strokeWidth="0.5"
                                    />
                                  )
                                })}
                                {actualPath && (
                                  <path
                                    d={actualPath}
                                    fill="none"
                                    stroke="#a5b4fc"
                                    strokeWidth="1.6"
                                  />
                                )}
                                {predictedPath && (
                                  <path
                                    d={predictedPath}
                                    fill="none"
                                    stroke="#22c55e"
                                    strokeWidth="1.6"
                                  />
                                )}
                              </g>
                            )
                          })()}
                        </svg>
                      </div>
                    </div>
                  )}

                  {forecast.forecast_plot && (
                    <img
                      className="forecast-plot"
                      src={`${API_BASE}/forecast/plot?ts=${Date.now()}`}
                      alt="Matplotlib forecast plot"
                    />
                  )}
                </>
              )}
            </div>
          </section>
        </div>

        <footer className="footer">
          React + Flask + XGBoost · Built for a hackathon-style time series forecasting demo.
        </footer>
      </div>
    </div>
  )
}

export default App

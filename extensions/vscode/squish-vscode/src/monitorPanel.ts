/**
 * squish-vscode/src/monitorPanel.ts
 *
 * WebviewView for the "Squish Monitor" activity-bar panel.
 * Polls the /health endpoint every 2 seconds and renders live metrics —
 * matching the monitoring dashboard look of the Squish web chat UI.
 */
import * as vscode from 'vscode';
import { SquishClient, HealthInfo } from './squishClient';

export class MonitorPanel implements vscode.WebviewViewProvider {
    public static readonly viewType = 'squish.monitorView';

    private _view?: vscode.WebviewView;
    private _pollTimer?: NodeJS.Timeout;
    private _sparkTps: number[] = [];           // last 30 tok/s samples
    private _sparkReq: number[] = [];           // last 30 req/s samples
    private _prevRequests = 0;
    private static readonly SPARK_LEN = 30;
    private static readonly POLL_MS   = 2_000;

    constructor(private readonly _extensionUri: vscode.Uri) {}

    // ── WebviewViewProvider ───────────────────────────────────────────────

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [vscode.Uri.joinPath(this._extensionUri, 'media')],
        };

        webviewView.webview.html = this._buildHtml(webviewView.webview);

        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                this._startPolling();
            } else {
                this._stopPolling();
            }
        });

        this._startPolling();
    }

    dispose(): void {
        this._stopPolling();
    }

    // ── Polling ───────────────────────────────────────────────────────────

    private _startPolling(): void {
        if (this._pollTimer) return;
        void this._poll();   // immediate first poll
        this._pollTimer = setInterval(() => { void this._poll(); }, MonitorPanel.POLL_MS);
    }

    private _stopPolling(): void {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = undefined;
        }
    }

    private async _poll(): Promise<void> {
        const cfg = vscode.workspace.getConfiguration('squish');
        const host: string  = cfg.get('host', '127.0.0.1');
        const port: number  = cfg.get('port', 11435);
        const apiKey: string = cfg.get('apiKey', 'squish');

        const client = new SquishClient(host, port, apiKey);

        let health: HealthInfo & { ram_gb?: number; gpu_pct?: number } = { loaded: false };
        try {
            health = await (client as unknown as { health(): Promise<typeof health> }).health() ?? { loaded: false };
        } catch {
            // server offline — send offline state
        }

        this._updateSparklines(health);
        this._view?.webview.postMessage({ type: 'monitorUpdate', health, sparkTps: [...this._sparkTps], sparkReq: [...this._sparkReq] });
    }

    private _updateSparklines(health: HealthInfo): void {
        const tps = health.tps ?? 0;
        this._sparkTps.push(tps);
        if (this._sparkTps.length > MonitorPanel.SPARK_LEN) this._sparkTps.shift();

        const req = health.requests ?? 0;
        const delta = Math.max(0, req - this._prevRequests);
        this._prevRequests = req;
        this._sparkReq.push(delta);
        if (this._sparkReq.length > MonitorPanel.SPARK_LEN) this._sparkReq.shift();
    }

    // ── HTML ──────────────────────────────────────────────────────────────

    private _buildHtml(webview: vscode.Webview): string {
        const nonce = _nonce();
        const csp   = `default-src 'none'; style-src 'nonce-${nonce}'; script-src 'nonce-${nonce}';`;

        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Squish Monitor</title>
  <style nonce="${nonce}">
    :root {
      --bg:        #0c0a14;
      --surface:   #131020;
      --surface2:  #1a1430;
      --border:    #2d1f4e;
      --accent:    #8B5CF6;
      --accent-pk: #EC4899;
      --text:      #e6edf3;
      --text-dim:  #8b949e;
      --success:   #3fb950;
      --warn:      #d29922;
      --danger:    #f85149;
    }
    * { box-sizing:border-box; margin:0; padding:0; }
    body {
      background:var(--bg);
      color:var(--text);
      font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
      font-size:12px;
      padding:12px 10px;
    }
    .status-badge {
      display:inline-flex; align-items:center; gap:6px;
      padding:4px 10px; border-radius:20px;
      font-size:11px; font-weight:600; letter-spacing:.4px;
      margin-bottom:14px; text-transform:uppercase;
    }
    .status-badge.online  { background:rgba(63,185,80,.15); color:var(--success); border:1px solid rgba(63,185,80,.3); }
    .status-badge.offline { background:rgba(248,81,73,.15); color:var(--danger);  border:1px solid rgba(248,81,73,.3); }
    .dot { width:6px; height:6px; border-radius:50%; }
    .dot.online  { background:var(--success); box-shadow:0 0 6px var(--success); }
    .dot.offline { background:var(--danger);  box-shadow:0 0 6px var(--danger); }

    .metrics-grid {
      display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;
    }
    .metric-card {
      background:var(--surface); border:1px solid var(--border);
      border-radius:10px; padding:10px 12px;
    }
    .metric-label {
      font-size:10px; color:var(--text-dim); text-transform:uppercase;
      letter-spacing:.5px; margin-bottom:4px;
    }
    .metric-value {
      font-size:18px; font-weight:700;
      background:linear-gradient(135deg,var(--accent),var(--accent-pk));
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .metric-unit { font-size:10px; color:var(--text-dim); }
    .metric-sub  { font-size:10px; color:var(--text-dim); margin-top:2px; }

    canvas { display:block; width:100%; height:32px; border-radius:4px; margin-top:6px; }

    .section-title {
      font-size:10px; color:var(--text-dim); text-transform:uppercase;
      letter-spacing:.6px; margin:12px 0 6px;
    }

    .model-tag {
      background:linear-gradient(135deg,rgba(139,92,246,.2),rgba(236,72,153,.2));
      border:1px solid var(--border); border-radius:8px;
      padding:6px 10px; font-size:11px; word-break:break-all;
      color:var(--text);
    }

    .console {
      background:var(--surface); border:1px solid var(--border);
      border-radius:8px; padding:8px 10px;
      font-family:'SF Mono','JetBrains Mono',monospace; font-size:10px;
      color:var(--text-dim); max-height:120px; overflow-y:auto;
      line-height:1.5;
    }
    .console-line.err  { color:var(--danger); }
    .console-line.warn { color:var(--warn); }
    .console-line.ok   { color:var(--success); }
  </style>
</head>
<body>
  <div id="badge" class="status-badge offline">
    <span class="dot offline" id="dot"></span>
    <span id="status-text">Offline</span>
  </div>

  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Throughput</div>
      <div><span class="metric-value" id="tps">—</span> <span class="metric-unit">tok/s</span></div>
      <canvas id="canvas-tps" width="120" height="32"></canvas>
    </div>
    <div class="metric-card">
      <div class="metric-label">Requests</div>
      <div><span class="metric-value" id="reqs">—</span></div>
      <div class="metric-sub">total served</div>
      <canvas id="canvas-req" width="120" height="32"></canvas>
    </div>
    <div class="metric-card">
      <div class="metric-label">Uptime</div>
      <div><span class="metric-value" id="uptime">—</span></div>
      <div class="metric-sub" id="uptime-unit">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">RAM</div>
      <div><span class="metric-value" id="ram">—</span> <span class="metric-unit">GB</span></div>
      <div class="metric-sub" id="gpu-pct">GPU: —%</div>
    </div>
  </div>

  <div class="section-title">Active Model</div>
  <div class="model-tag" id="model-tag">—</div>

  <div class="section-title" style="margin-top:16px">Console</div>
  <div class="console" id="console"></div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();

    // ── Sparkline helpers ───────────────────────────────────────────────

    function drawSparkline(canvasId, data, color) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0,0,W,H);
      if (!data || data.length < 2) return;
      const max = Math.max(...data, 1);
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = (i / (data.length-1)) * W;
        const y = H - (v / max) * H * 0.85 - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // fill area
      ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = color.replace(')', ',0.15)').replace('rgb', 'rgba');
      ctx.fill();
    }

    // ── Message handler ─────────────────────────────────────────────────

    window.addEventListener('message', (e) => {
      const { type, health, sparkTps, sparkReq } = e.data;
      if (type !== 'monitorUpdate') return;

      const online = health.loaded === true;
      const badge  = document.getElementById('badge');
      const dot    = document.getElementById('dot');
      const st     = document.getElementById('status-text');

      badge.className  = 'status-badge ' + (online ? 'online' : 'offline');
      dot.className    = 'dot '          + (online ? 'online' : 'offline');
      st.textContent   = online ? 'Online' : 'Offline';

      document.getElementById('tps').textContent =
        health.tps != null ? health.tps.toFixed(1) : '—';
      document.getElementById('reqs').textContent =
        health.requests != null ? String(health.requests) : '—';

      const upSec = health.uptime ?? 0;
      let upStr, upUnit;
      if (upSec < 60)       { upStr = String(Math.round(upSec));         upUnit = 'seconds'; }
      else if (upSec < 3600){ upStr = (upSec/60).toFixed(1);             upUnit = 'minutes'; }
      else                  { upStr = (upSec/3600).toFixed(1);           upUnit = 'hours'; }
      document.getElementById('uptime').textContent      = upStr;
      document.getElementById('uptime-unit').textContent = upUnit;

      document.getElementById('ram').textContent     = health.ram_gb  != null ? health.ram_gb.toFixed(1)  : '—';
      document.getElementById('gpu-pct').textContent = health.gpu_pct != null ? ('GPU: ' + health.gpu_pct + '%') : 'GPU: —%';
      document.getElementById('model-tag').textContent = health.model ?? '—';

      drawSparkline('canvas-tps', sparkTps, 'rgb(139,92,246)');
      drawSparkline('canvas-req', sparkReq, 'rgb(236,72,153)');

      // Console log
      const cons = document.getElementById('console');
      const line = document.createElement('div');
      const ts   = new Date().toLocaleTimeString();
      line.className = 'console-line ' + (online ? 'ok' : 'err');
      line.textContent = ts + '  ' + (online
        ? ('model=' + (health.model||'?') + '  tps=' + (health.tps||0).toFixed(1) + '  req=' + (health.requests||0))
        : 'server offline');
      cons.appendChild(line);
      // Keep last 50 lines
      while (cons.children.length > 50) cons.removeChild(cons.firstChild);
      cons.scrollTop = cons.scrollHeight;
    });
  </script>
</body>
</html>`;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

function _nonce(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < 32; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

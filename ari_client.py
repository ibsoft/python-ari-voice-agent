import base64, json, logging, threading, time, urllib.parse
import requests
import websocket

log = logging.getLogger("ari_client")

class AriClient:
    def __init__(self, host, port, username, password, app):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.app = app
        self._ws = None
        self._handlers = {}

        auth = f"{username}:{password}".encode()
        self._auth_header = "Basic " + base64.b64encode(auth).decode()

    def _base_http(self):
        return f"http://{self.host}:{self.port}/ari"

    def _base_ws(self):
        return f"ws://{self.host}:{self.port}/ari"

    def on(self, event_name, handler):
        self._handlers.setdefault(event_name, []).append(handler)

    def _emit(self, evt):
        name = evt.get("type") or evt.get("event")
        for h in self._handlers.get(name, []):
            try:
                h(evt)
            except Exception as ex:
                log.exception("handler error: %s", ex)

    def start_events(self):
        params = urllib.parse.urlencode({"app": self.app})
        url = f"{self._base_ws()}/events?{params}"
        headers = [f"Authorization: {self._auth_header}"]

        def on_message(ws, message):
            try:
                evt = json.loads(message)
                self._emit(evt)
            except Exception as ex:
                log.error("Event parse error: %s", ex)

        def on_error(ws, error):
            log.error("WS error: %s", error)

        def on_close(ws, code, msg):
            log.info("WS closed: %s %s", code, msg)

        self._ws = websocket.WebSocketApp(url, header=headers, on_message=on_message, on_error=on_error, on_close=on_close)
        t = threading.Thread(target=self._ws.run_forever, daemon=True)
        t.start()

    # ---- REST helpers ----
    def _post(self, path, params=None, json_body=None):
        url = f"{self._base_http()}{path}"
        headers = {"Authorization": self._auth_header}
        r = requests.post(url, params=params, json=json_body, headers=headers, timeout=10)
        r.raise_for_status()
        if r.content:
            return r.json()
        return {}

    def _delete(self, path, params=None):
        url = f"{self._base_http()}{path}"
        headers = {"Authorization": self._auth_header}
        r = requests.delete(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        if r.content:
            return r.json()
        return {}

    # Channels
    def channels_answer(self, channel_id):
        return self._post(f"/channels/{channel_id}/answer")

    def channels_external_media(self, external_host, fmt="slin16", direction="both"):
        params = {"app": self.app, "external_host": external_host, "format": fmt, "direction": direction}
        return self._post("/channels/externalMedia", params=params)

    # Bridges
    def bridges_create(self, types="mixing,dtmf_events"):
        return self._post("/bridges", params={"type": types})

    def bridges_add_channel(self, bridge_id, channel_id):
        return self._post(f"/bridges/{bridge_id}/addChannel", params={"channel": channel_id})

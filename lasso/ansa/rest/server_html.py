SERVER_HTML_TEMPLATE = '''
<!DOCTYPE html> 
<html>
    <head>
        <title>ANSA REST</title>
        <meta http-equiv="Content-type" content="text/html;charset=UTF-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.4/ace.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/ace/1.4.4/mode-json.js"></script>

        <style>{materialize_css}</style>
        <style>
        
            /* MATERIALIZE THEME*/
            .dark-primary-color    {{ background: #263238; }}
            .default-primary-color {{ background: #455A64; }}
            .light-primary-color   {{ background: #90A4AE; }}
            .accent-color          {{ background: #00B0FF; }}
            .accent-light          {{ background: #40C4FF; }}
            .accent-tint           {{ background: #33C0FF; }}
            .accent-shade          {{ background: #009EEB; }}
            .accent-button       {{ background: #00B0FF; box-shadow: 0px 1px 0px #33C0FF inset, 0px -1px 0px #009EEB inset, 0px 0px 5px rgba(0, 0, 0, 0.5); }}

            .center-items-flex {{
                display:flex;
                flex-direction: row;
                justify-content: center;
                align-items: center;
            }}

            .main-container {{
                margin:auto;
                width: 75%;
            }}

            body {{
                color:#e1e1e1;
                min-height: 100vh;
                padding: 0;
                margin: 0;
                background: linear-gradient(to bottom, #000 0%, #1a2028 50%, #293845 100%);
            }}

            .btn, .btn-large, .btn-small {{
                background-color: #2e8aca;
            }}

            .btn:hover, .btn-large:hover, .btn-small:hover {{
                background-color: #2e45a8;
            }}

            .btn:focus, .btn-large:focus, .btn-small:focus, .btn-floating:focus {{
                background-color: #2e45a8;
            }}

            button:focus {{
                background-color: #2e45a8;
            }}

            .border-slight-blue {{
                border: 1px solid #2786c74d;
            }}

        </style>
    </head>

    <body >

        <div class="main-container">

            <!-- HEADER -->
            <div style="display:flex; flex-direction: column; align-items:center; padding-top: 35px;">
                <a href="https://www.lasso.de">
                    <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjxzdmcKICAgeG1sbnM6ZGM9Imh0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvIgogICB4bWxuczpjYz0iaHR0cDovL2NyZWF0aXZlY29tbW9ucy5vcmcvbnMjIgogICB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiCiAgIHhtbG5zOnNvZGlwb2RpPSJodHRwOi8vc29kaXBvZGkuc291cmNlZm9yZ2UubmV0L0RURC9zb2RpcG9kaS0wLmR0ZCIKICAgeG1sbnM6aW5rc2NhcGU9Imh0dHA6Ly93d3cuaW5rc2NhcGUub3JnL25hbWVzcGFjZXMvaW5rc2NhcGUiCiAgIGhlaWdodD0iNTMuODM0NzI4IgogICB3aWR0aD0iMTcxLjYwODgxIgogICB2ZXJzaW9uPSIxLjEiCiAgIHZpZXdCb3g9IjAgMCAxNzEuNjA4ODEgNTMuODM0NzI4IgogICBkYXRhLW5hbWU9IkViZW5lIDEiCiAgIGlkPSJFYmVuZV8xIgogICBpbmtzY2FwZTp2ZXJzaW9uPSIwLjkxIHIxMzcyNSIKICAgc29kaXBvZGk6ZG9jbmFtZT0ibGFzc29fbG9nbzEuc3ZnIj4KICA8c29kaXBvZGk6bmFtZWR2aWV3CiAgICAgcGFnZWNvbG9yPSIjZmZmZmZmIgogICAgIGJvcmRlcmNvbG9yPSIjNjY2NjY2IgogICAgIGJvcmRlcm9wYWNpdHk9IjEiCiAgICAgb2JqZWN0dG9sZXJhbmNlPSIxMCIKICAgICBncmlkdG9sZXJhbmNlPSIxMCIKICAgICBndWlkZXRvbGVyYW5jZT0iMTAiCiAgICAgaW5rc2NhcGU6cGFnZW9wYWNpdHk9IjAiCiAgICAgaW5rc2NhcGU6cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTp3aW5kb3ctd2lkdGg9IjE3NDMiCiAgICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iOTc3IgogICAgIGlkPSJuYW1lZHZpZXcxMyIKICAgICBzaG93Z3JpZD0iZmFsc2UiCiAgICAgZml0LW1hcmdpbi10b3A9IjEiCiAgICAgZml0LW1hcmdpbi1sZWZ0PSIxIgogICAgIGZpdC1tYXJnaW4tcmlnaHQ9IjEiCiAgICAgZml0LW1hcmdpbi1ib3R0b209IjEiCiAgICAgaW5rc2NhcGU6em9vbT0iNS4wMDU2MTI3IgogICAgIGlua3NjYXBlOmN4PSI3OS4wOTkyNDgiCiAgICAgaW5rc2NhcGU6Y3k9IjguMTE5MDAwNiIKICAgICBpbmtzY2FwZTp3aW5kb3cteD0iNjI5IgogICAgIGlua3NjYXBlOndpbmRvdy15PSIzMjgiCiAgICAgaW5rc2NhcGU6d2luZG93LW1heGltaXplZD0iMCIKICAgICBpbmtzY2FwZTpjdXJyZW50LWxheWVyPSJFYmVuZV8xIiAvPgogIDxtZXRhZGF0YQogICAgIGlkPSJtZXRhZGF0YTcwIj4KICAgIDxyZGY6UkRGPgogICAgICA8Y2M6V29yawogICAgICAgICByZGY6YWJvdXQ9IiI+CiAgICAgICAgPGRjOmZvcm1hdD5pbWFnZS9zdmcreG1sPC9kYzpmb3JtYXQ+CiAgICAgICAgPGRjOnR5cGUKICAgICAgICAgICByZGY6cmVzb3VyY2U9Imh0dHA6Ly9wdXJsLm9yZy9kYy9kY21pdHlwZS9TdGlsbEltYWdlIiAvPgogICAgICAgIDxkYzp0aXRsZT5aZWljaGVuZmzDpGNoZSAxPC9kYzp0aXRsZT4KICAgICAgPC9jYzpXb3JrPgogICAgPC9yZGY6UkRGPgogIDwvbWV0YWRhdGE+CiAgPGRlZnMKICAgICBpZD0iZGVmczEzIj4KICAgIDxzdHlsZQogICAgICAgaWQ9InN0eWxlMiI+LmNscy0xe2ZpbGw6dXJsKCNVbmJlbmFubnRlcl9WZXJsYXVmXzcpO30uY2xzLTJ7ZmlsbDojMzI4Y2NjO308L3N0eWxlPgogICAgPGxpbmVhckdyYWRpZW50CiAgICAgICBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKC0xMiwtMC4xOCkiCiAgICAgICBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIKICAgICAgIHkyPSItMjEuMDciCiAgICAgICB4Mj0iMTQ5Ljc4IgogICAgICAgeTE9IjczLjQwMDAwMiIKICAgICAgIHgxPSI1NS4zMiIKICAgICAgIGlkPSJVbmJlbmFubnRlcl9WZXJsYXVmXzciPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDQiCiAgICAgICAgIHN0b3AtY29sb3I9IiMwMDZlYjciCiAgICAgICAgIG9mZnNldD0iMCIgLz4KICAgICAgPHN0b3AKICAgICAgICAgaWQ9InN0b3A2IgogICAgICAgICBzdG9wLWNvbG9yPSIjMTE3OGJlIgogICAgICAgICBvZmZzZXQ9IjAuMjYiIC8+CiAgICAgIDxzdG9wCiAgICAgICAgIGlkPSJzdG9wOCIKICAgICAgICAgc3RvcC1jb2xvcj0iIzI5ODdjOCIKICAgICAgICAgb2Zmc2V0PSIwLjcyIiAvPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDEwIgogICAgICAgICBzdG9wLWNvbG9yPSIjMzI4Y2NjIgogICAgICAgICBvZmZzZXQ9IjEiIC8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50CiAgICAgICBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKC0yNy4xODI5Niw3MS45OTQzMTcpIgogICAgICAgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiCiAgICAgICB5Mj0iLTIxLjA3IgogICAgICAgeDI9IjE0OS43OCIKICAgICAgIHkxPSI3My40MDAwMDIiCiAgICAgICB4MT0iNTUuMzIiCiAgICAgICBpZD0iVW5iZW5hbm50ZXJfVmVybGF1Zl83LTMiPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDQtNiIKICAgICAgICAgc3RvcC1jb2xvcj0iIzAwNmViNyIKICAgICAgICAgb2Zmc2V0PSIwIiAvPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDYtNyIKICAgICAgICAgc3RvcC1jb2xvcj0iIzExNzhiZSIKICAgICAgICAgb2Zmc2V0PSIwLjI2IiAvPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDgtNSIKICAgICAgICAgc3RvcC1jb2xvcj0iIzI5ODdjOCIKICAgICAgICAgb2Zmc2V0PSIwLjcyIiAvPgogICAgICA8c3RvcAogICAgICAgICBpZD0ic3RvcDEwLTMiCiAgICAgICAgIHN0b3AtY29sb3I9IiMzMjhjY2MiCiAgICAgICAgIG9mZnNldD0iMSIgLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8bGluZWFyR3JhZGllbnQKICAgICAgIGlua3NjYXBlOmNvbGxlY3Q9ImFsd2F5cyIKICAgICAgIHhsaW5rOmhyZWY9IiNVbmJlbmFubnRlcl9WZXJsYXVmXzciCiAgICAgICBpZD0ibGluZWFyR3JhZGllbnQ0MTgxIgogICAgICAgeDE9IjEwLjI3NDM5MyIKICAgICAgIHkxPSI1NC44NzUzNDciCiAgICAgICB4Mj0iMTc5LjYwODgiCiAgICAgICB5Mj0iNTQuODc1MzQ3IgogICAgICAgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiCiAgICAgICBncmFkaWVudFRyYW5zZm9ybT0idHJhbnNsYXRlKC05LC05KSIgLz4KICAgIDxsaW5lYXJHcmFkaWVudAogICAgICAgaW5rc2NhcGU6Y29sbGVjdD0iYWx3YXlzIgogICAgICAgeGxpbms6aHJlZj0iI1VuYmVuYW5udGVyX1ZlcmxhdWZfNyIKICAgICAgIGlkPSJsaW5lYXJHcmFkaWVudDQxODMiCiAgICAgICBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIKICAgICAgIGdyYWRpZW50VHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTIxLC05LjE4KSIKICAgICAgIHgxPSI1NS4zMiIKICAgICAgIHkxPSI3My40MDAwMDIiCiAgICAgICB4Mj0iMTQ5Ljc4IgogICAgICAgeTI9Ii0yMS4wNyIgLz4KICA8L2RlZnM+CiAgPHRpdGxlCiAgICAgaWQ9InRpdGxlMTUiPlplaWNoZW5mbMOkY2hlIDE8L3RpdGxlPgogIDxwYXRoCiAgICAgc3R5bGU9ImZpbGw6dXJsKCNsaW5lYXJHcmFkaWVudDQxODMpIgogICAgIGlkPSJwYXRoMTciCiAgICAgZD0ibSAxNTMsMSAtMzguMzksMCAtNDAuMiwwIGEgOC41LDguNSAwIDEgMCAwLDE3IGwgNC45NCwwIGEgNy4wOSw3LjA5IDAgMCAxIDAsMTQuMTcgbCAtMzkuNTMsMCBhIDIuNSwyLjUgMCAxIDEgMCwtNSBsIDEyLDAgYSAzLjg5LDMuODkgMCAwIDAgMy42NSwtNS4yMiBMIDQ5LjI1LDQuODIgYSA1LjY0LDUuNjQgMCAwIDAgLTEwLjYyLDAgbCAtOC45NSwyNC41OSBhIDQuMjUsNC4yNSAwIDAgMSAtNCwyLjggbCAtMTcuNjIsMCBBIDQuMjUsNC4yNSAwIDAgMSAzLjgzLDI3Ljk2IGwgMCwtMjYuOTIgLTIuODMsMCAwLDI2LjkyIGEgNy4wOCw3LjA4IDAgMCAwIDcuMDYsNy4wOCBsIDE3LjY1LDAgYSA3LjA4LDcuMDggMCAwIDAgNi42NCwtNC42NiBMIDQxLjMsNS43NCBhIDIuODIsMi44MiAwIDAgMSA1LjMsMCBMIDUyLjgzLDIzIGEgMS4wNywxLjA3IDAgMCAxIC0xLDEuNDMgbCAtMTIsMCBhIDUuMzEsNS4zMSAwIDAgMCAwLDEwLjYyIGwgNzQuODIsMCBhIDkuOTE1LDkuOTE1IDAgMCAwIDAsLTE5LjgzIGwgLTQuOTQsMCBhIDUuNjcsNS42NyAwIDEgMSAwLC0xMS4zNCBsIDMzLjkzLDAgYSAxNi45MiwxNi45MiAwIDEgMCA5LjQsLTIuODcgeiBtIC00My4zMywxNyA0Ljk0LDAgYSA3LjA5LDcuMDkgMCAwIDEgMCwxNC4xNyBsIC0yOC40LDAgYSA5LjkxLDkuOTEgMCAwIDAgLTYuOSwtMTcgbCAtNC45LDAgYSA1LjY3LDUuNjcgMCAxIDEgMCwtMTEuMzQgbCAyOSwwIEEgOC40OSw4LjQ5IDAgMCAwIDEwOS43LDE4IFogTSAxNTMsMzIuMTcgYSAxNC4xNiwxNC4xNiAwIDAgMSAtMC43NywtMjguMyBsIDAuNzksMCAwLDAgYSAxNC4xNywxNC4xNyAwIDAgMSAwLDI4LjM0IHoiCiAgICAgY2xhc3M9ImNscy0xIgogICAgIGlua3NjYXBlOmNvbm5lY3Rvci1jdXJ2YXR1cmU9IjAiIC8+CiAgPHRleHQKICAgICB4bWw6c3BhY2U9InByZXNlcnZlIgogICAgIHN0eWxlPSJmb250LXN0eWxlOm5vcm1hbDtmb250LXdlaWdodDpub3JtYWw7Zm9udC1zaXplOjQwcHg7bGluZS1oZWlnaHQ6MTI1JTtmb250LWZhbWlseTpzYW5zLXNlcmlmO2xldHRlci1zcGFjaW5nOjBweDt3b3JkLXNwYWNpbmc6MHB4O2ZpbGw6dXJsKCNsaW5lYXJHcmFkaWVudDQxODEpO2ZpbGwtb3BhY2l0eToxO3N0cm9rZTpub25lO3N0cm9rZS13aWR0aDoxcHg7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW9wYWNpdHk6MSIKICAgICB4PSItMC4wMTAwOTE3ODIiCiAgICAgeT0iNDkuODQ0NjE2IgogICAgIGlkPSJ0ZXh0NDE3MSIKICAgICBzb2RpcG9kaTpsaW5lc3BhY2luZz0iMTI1JSI+PHRzcGFuCiAgICAgICBzb2RpcG9kaTpyb2xlPSJsaW5lIgogICAgICAgaWQ9InRzcGFuNDE3MyIKICAgICAgIHg9Ii0wLjAxMDA5MTc4MiIKICAgICAgIHk9IjQ5Ljg0NDYxNiIKICAgICAgIHN0eWxlPSJmb250LXNpemU6MTQuMzc1cHg7ZmlsbDp1cmwoI2xpbmVhckdyYWRpZW50NDE4MSk7ZmlsbC1vcGFjaXR5OjEiPkluZ2VuaWV1cmdlc2VsbHNjaGFmdCBtYkg8L3RzcGFuPjwvdGV4dD4KPC9zdmc+Cg==" />
                </a>
                <h2 style="margin-left: 1em;">ANSA Rest Server</h2>
            </div>

            <hr>

            <!-- INFO -->
            <p>
                <h4>Server Info:</h4>

                <table style="width: unset;">
                    <tr>
                        <td>Address</td>
                        <td><a href="{address}">{address}</a></td>
                    </tr>
                    <tr>
                        <td>Address for Rest API (POST)</td>
                        <td><a href="{address_run}">{address_run}</a></td>
                    </tr>
                    <tr>
                        <td>Shutdown service</td>
                        <td><a href="{address_shutdown}">{address_shutdown}</a></td>
                    </tr>
                </table>
            </p>

            <hr>

            <p>
                <h4>Testing:</h4>
                <p>
                    Test section for running an ANSA function through REST:
                </p>

                <div style="display:flex; flex-direction: row; min-height:600px;">
                        <div id="editor" class="border-slight-blue" style="flex:1; margin-right: 1rem;">{{
    "function_name": "ansa.base.CreateEntity",
    "args": [
    ],
    "kwargs": {{
        "deck": 1,
        "element_type": "POINT"
    }}
}}</div>
                        <button style="margin: auto;" class="btn" onclick="onButtonSend()">
                            Send
                        </button>
                        <div class="border-slight-blue" id="editor-response" style="flex:1; margin-left: 1rem;"></div>

                </div>
            </p>

        </div>


        <!-- MATERIALIZE -->
        <script>{materialize_js}</script>
        <!-- EDITOR -->
        <script>
            ace.config.set('basePath', 'hihihi')

            // editor for the input
            document.getElementById("editor").style.fontSize='1.1rem';
            let editor = ace.edit("editor");
            editor.setTheme("ace/theme/monokai");
            editor.session.setMode("ace/mode/json");
            editor.getSession().setTabSize(2);
            editor.getSession().setUseWrapMode(true);

            // editor for the response
            document.getElementById("editor-response").style.fontSize='1.1rem';
            let editorResponse = ace.edit("editor-response");
            editorResponse.setTheme("ace/theme/monokai");
            editorResponse.session.setMode("ace/mode/json");
            editorResponse.getSession().setTabSize(2);
            editorResponse.getSession().setUseWrapMode(true);

            // for run requests
            let xhttp = new XMLHttpRequest();

            // what happens if we receive something?
            xhttp.onreadystatechange = function() {{
                if (this.readyState == 4 && this.status == 200) {{
                    console.log("Receiving",this.responseText);
                    // editor_response.set(JSON.parse(this.responseText));
                    try {{
                        const data = JSON.parse(this.responseText);
                        const niceJson = JSON.stringify(data, null, "  ");
                        editorResponse.setValue(niceJson);
                    }} catch (error) {{
                        editorResponse.setValue(this.responseText);
                    }}
                    editorResponse.gotoLine(0);
                }}
            }};

            // triggers the send function
            function onButtonSend(){{
                xhttp.open("POST", "{address_run}", true);
                xhttp.setRequestHeader("Content-type", "application/json");
                xhttp.send(editor.getValue());
            }}
        </script>
    </body>
</html>
'''
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.staticfiles import StaticFiles

from graphs.graph import LangGraphProcess

from api.routes.chat import router as chat_router
from api.routes.ping import router as ping_router
from azure_agent.api.routes.delete_thread import router as thread_router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""

    # Initialize LangGraph Agent
    agent = LangGraphProcess()
    if getattr(agent, "graph", None) is None:
        await agent.setup()

    # App State Initialization
    app.state.agent = agent
    logger.info("[app.py] Agent Initialize Success")

    # TTL sweeper (store)
    store = getattr(agent, "store", None)
    if store is not None:
        start = getattr(store, "start_ttl_sweeper", None)
        if callable(start):
            try:
                await start(sweep_interval_minutes=60)
                logger.info("[app.py] Success to start Postgres TTL sweeper")
            except Exception as exc:
                logger.warning("[app.py] Failed to start Postgres TTL sweeper: %s", exc)

    # Yield to application
    try:
        yield
    
    # Application Shutdown
    finally:
        agent = getattr(app.state, "agent", None)
        app.state.agent = None

        if agent is not None:
            try:
                await agent.close()
            except Exception as exc:
                logger.warning("[app.py] Application Cleanup Failed: %s", exc)

        logger.info("[app.py] Application shutdown")


def create_app() -> FastAPI:
    """
    Create FastAPI application instance

    API Endpoints:
        - /agent/ping: Health check endpoint
        - /agent/chat: Chat interaction endpoint
        - /agent/delete_thread: Delete Thread endpoint
    """

    # FastAPI Instance
    app = FastAPI(
        title="Azure Agent API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        openapi_url="/agent/openapi.json",
        redoc_url=None,
        swagger_ui_oauth2_redirect_url="/agent/swagger/oauth2-redirect",
    )

    # Swagger UI (Static Files)
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/agent/static", StaticFiles(directory=static_dir), name="static")

    # Swagger UI (Endpoints)
    @app.get("/agent/swagger", include_in_schema=False)
    async def custom_swagger_ui_html(request: Request):
        return get_swagger_ui_html(
            openapi_url=request.url_for("openapi").path,
            title="Azure Agent API",
            oauth2_redirect_url=request.url_for("swagger_ui_redirect").path,
            swagger_js_url=request.url_for("static", path="swagger-ui-bundle.js").path,
            swagger_css_url=request.url_for("static", path="swagger-ui.css").path,
            swagger_favicon_url=request.url_for("static", path="favicon.png").path,
        )

    # Swagger UI (OAuth2 Redirect) 
    @app.get("/agent/swagger/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    # FastAPI (CORS Setting)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:7860",
            "http://127.0.0.1:7860",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routes
    app.include_router(ping_router)
    app.include_router(chat_router)
    app.include_router(thread_router)

    return app

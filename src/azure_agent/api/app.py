import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi.staticfiles import StaticFiles

from infra.key_vault import create_secret_client
from infra.redis import close_redis_client, create_redis_stream_client

from api.routes.job import router as job_router
from api.routes.ping import router as ping_router

logger = logging.getLogger(__name__)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Initialize Secret Client
    secret_client = create_secret_client()

    # Initialize Redis Stream Client (queue)
    redis_stream_client = None
    try:
        redis_stream_client = create_redis_stream_client(secret_client)
        app.state.redis_stream_client = redis_stream_client
        logger.info("[app.py] Stream Redis Initialize Success")
    except Exception as exc:
        app.state.redis_stream_client = None
        logger.exception("[app.py] Failed to initialize stream redis client")
        raise RuntimeError("Stream Redis initialization failed") from exc

    # Yield to application
    try:
        yield
    
    # Application Shutdown
    finally:
        redis_stream_client = getattr(app.state, "redis_stream_client", None)
        app.state.redis_stream_client = None

        if redis_stream_client is not None:
            await close_redis_client(redis_stream_client)

        logger.info("[app.py] Application shutdown")


def create_app() -> FastAPI:
    """
    Create FastAPI application instance

    API Endpoints:
        - /agent/ping: Health check endpoint
        - /agent/jobs: Async job endpoints
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
    app.include_router(job_router)

    return app

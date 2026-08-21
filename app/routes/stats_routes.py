from flask import Blueprint, render_template
from app.services.stats_service import get_performance_metrics

stats_bp = Blueprint("stats", __name__)


@stats_bp.route("/stats", methods=["GET"])
def stats():
    metrics = get_performance_metrics()
    return render_template("stats.html", metrics=metrics)

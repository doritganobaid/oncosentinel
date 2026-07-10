from __future__ import annotations

from oncosentinel.execute.tool_contraindication import check_contraindication
from oncosentinel.execute.tool_dgidb import query_dgidb
from oncosentinel.execute.tool_dispatch import Dispatcher
from oncosentinel.execute.tool_mcid import lookup_mcid
from oncosentinel.execute.tool_pareto import compute_pareto_frontier


def build_dispatcher(max_retries: int = 1) -> Dispatcher:
    dispatcher = Dispatcher(max_retries=max_retries)
    dispatcher.register("query_DGIdb", query_dgidb)
    dispatcher.register("lookup_MCID", lookup_mcid)
    dispatcher.register("check_contraindication", check_contraindication)
    dispatcher.register("compute_pareto_frontier", compute_pareto_frontier)
    return dispatcher

"""Keep a trusted search stop out of upstream candidate-error retry handlers."""


class SearchResourceStop(BaseException):
    """Control-flow signal, not an algorithm failure or metric.

    OpenEvolve catches Exception in its evaluator, iteration worker and parent
    loop. A separate BaseException survives those boundaries and process pickle;
    its native run() finally still shuts down the process pool. Only trusted
    admission code raises this signal. Do not translate arbitrary exceptions.
    """


async def run_to_resource_boundary(evolution, **kwargs):
    try:
        best = await evolution.run(**kwargs)
        return {"status": "NATIVE_COMPLETE", "best": best, "reason": None}
    except SearchResourceStop as stop:
        # Same native selection as OpenEvolve.run, never select a different
        # winner by a post-hoc metric. Worker cleanup has already run in finally.
        database = evolution.database
        best = database.get(database.best_program_id) if database.best_program_id else None
        if best is None:
            best = database.get_best_program()
        evolution._save_checkpoint(database.last_iteration)
        if best is not None:
            evolution._save_best_program(best)
        return {"status": "RESOURCE_STOP", "best": best, "reason": str(stop)}

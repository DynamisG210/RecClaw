def install_api_policy():
    from main_api_ledger import MainApiLedger
    if getattr(MainApiLedger, '_current_resource_policy', False):
        return
    summary0, cost0 = MainApiLedger._summarize, MainApiLedger._cost_totals

    def admission_events(events):
        # Keep genuine successes even if a reconciled call also has a FAILED row.
        succeeded = {e['call_id'] for e in events if e['event'] == 'SUCCESS'}
        failed = {e['call_id'] for e in events if e['event'] == 'FAILED'} - succeeded
        return [e for e in events if e.get('call_id') not in failed]

    def summarize(events):
        return summary0(admission_events(events))

    def costs(self, events):
        return cost0(self, admission_events(events))

    def admit(self):
        with self._lock:
            input_cost, output_cost, _ = self._cost_totals(self.events())
            return input_cost + output_cost <= self.normalized_unit_usd * self.max_normalized_u

    MainApiLedger._summarize = staticmethod(summarize)
    MainApiLedger._cost_totals = costs
    MainApiLedger.admit_output = admit
    MainApiLedger._current_resource_policy = True

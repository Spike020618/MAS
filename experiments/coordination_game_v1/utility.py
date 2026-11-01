def utility(value,configuration,member):
    return value(configuration)-value(tuple(a for a in configuration if a.member_id!=member))

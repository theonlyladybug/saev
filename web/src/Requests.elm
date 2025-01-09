module Requests exposing
    ( Id
    , Result
    , State(..)
    , init
    , isStale
    , next
    )

-- MODEL


type Id
    = Id Int


init : Id
init =
    Id 0


type alias Result a =
    { id : Id
    , data : a
    }



-- UPDATE


next : Id -> Id
next (Id id) =
    Id (id + 1)


isStale : Id -> Id -> Bool
isStale (Id id) (Id last) =
    id < last



-- LoadingState


type State a
    = Initial
    | Loading
    | Loaded a
    | Failed String

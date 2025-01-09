module Comparison exposing (..)

import Browser
import Browser.Navigation
import Gradio
import Html
import Url


main =
    Browser.application
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        , onUrlRequest = onUrlRequest
        , onUrlChange = onUrlChange
        }



-- MESSAGE


type Msg
    = NoOp



-- MODEL


type alias Model =
    { -- Browser
      key : Browser.Navigation.Key

    -- APIs
    , gradio : Gradio.Config
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        model =
            { -- Browser
              key = key
            , gradio =
                { host = "https://localhost:7860" }
            }
    in
    ( model, Cmd.none )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    NoOp



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Image Classification"
    , body =
        [ Html.p []
            [ Html.text "Hello World" ]
        ]
    }

module Comparison exposing (..)

import Browser
import Browser.Navigation
import Dict
import File
import Gradio
import Html
import Html.Attributes
import Json.Decode as D
import Json.Encode as E
import Requests
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
    | GotImage Requests.Id (Result Gradio.Error Example)
    | GotSaeActivations Requests.Id (Result Gradio.Error (Dict.Dict String (List SaeActivation)))



-- MODEL


type alias Model =
    { -- Browser
      key : Browser.Navigation.Key

    -- State
    , inputExample : Requests.Requested Example

    -- Mapping from ViT name (string) to a list of activations. Changes as the input image changes.
    , saeActivations : Requests.Requested (Dict.Dict String (List SaeActivation))

    -- APIs
    , gradio : Gradio.Config
    , imageRequestId : Requests.Id
    , saeActivationsRequestId : Requests.Id
    }


type alias SaeActivation =
    { latent : Int
    , activations : List Float -- Has nPatches floats

    -- TODO: add examples
    , examples : List HighlightedExample
    }


type alias Example =
    { url : Gradio.Base64Image
    , label : String
    }


type alias HighlightedExample =
    { original : Gradio.Base64Image
    , highlighted : Gradio.Base64Image
    , label : String
    }


init : () -> Url.Url -> Browser.Navigation.Key -> ( Model, Cmd Msg )
init _ url key =
    let
        model =
            { -- Browser
              key = key
            , inputExample = Requests.Initial
            , saeActivations = Requests.Initial

            -- APIs
            , gradio =
                { host = "http://127.0.0.1:7860" }
            , imageRequestId = Requests.init
            , saeActivationsRequestId = Requests.init
            }
    in
    ( model, getImage model.gradio model.imageRequestId 0 )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        NoOp ->
            ( model, Cmd.none )

        GotImage id result ->
            if Requests.isStale id model.imageRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok example ->
                        let
                            saeActivationsRequestId =
                                Requests.next model.saeActivationsRequestId
                        in
                        ( { model
                            | inputExample = Requests.Loaded example
                            , saeActivationsRequestId = saeActivationsRequestId
                            , saeActivations = Requests.Loading
                          }
                        , getSaeActivations model.gradio
                            saeActivationsRequestId
                            example.url
                        )

                    Err err ->
                        ( { model
                            | inputExample = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )

        GotSaeActivations id result ->
            if Requests.isStale id model.saeActivationsRequestId then
                ( model, Cmd.none )

            else
                case result of
                    Ok saeActivations ->
                        ( { model
                            | saeActivations = Requests.Loaded saeActivations
                          }
                        , Cmd.none
                        )

                    Err err ->
                        ( { model
                            | saeActivations = Requests.Failed (explainGradioError err)
                          }
                        , Cmd.none
                        )


onUrlRequest : Browser.UrlRequest -> Msg
onUrlRequest request =
    NoOp


onUrlChange : Url.Url -> Msg
onUrlChange url =
    NoOp



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


updateIds : String -> Dict.Dict String Requests.Id -> ( Requests.Id, Dict.Dict String Requests.Id )
updateIds path ids =
    let
        id =
            Dict.get path ids |> nextId
    in
    ( id, Dict.insert path id ids )


nextId : Maybe Requests.Id -> Requests.Id
nextId maybeId =
    maybeId
        |> Maybe.withDefault Requests.init
        |> Requests.next


getImage : Gradio.Config -> Requests.Id -> Int -> Cmd Msg
getImage cfg id imageIndex =
    Gradio.get cfg
        "get-image"
        [ E.int imageIndex ]
        (D.map2 Example
            (D.index 0 Gradio.base64ImageDecoder)
            (D.index 1 D.string)
        )
        (GotImage id)


getSaeActivations : Gradio.Config -> Requests.Id -> Gradio.Base64Image -> Cmd Msg
getSaeActivations cfg id image =
    let
        latents =
            E.object
                [ ( "bioclip/inat21"
                  , E.list E.int
                        [ 449, 451, 518, 18380, 10185, 20085, 5435, 16081 ]
                  )
                , ( "clip/inat21"
                  , E.list E.int
                        [ 4661, 565, 5317 ]
                  )
                ]
    in
    Gradio.get cfg
        "get-sae-activations"
        [ Gradio.encodeImg image, latents ]
        (Gradio.decodeOne
            (D.dict
                (D.list
                    (D.map3
                        SaeActivation
                        (D.field "latent" D.int)
                        (D.field "activations" (D.list D.float))
                        (D.field "examples" (D.list highlightedExampleDecoder))
                    )
                )
            )
        )
        (GotSaeActivations id)


highlightedExampleDecoder : D.Decoder HighlightedExample
highlightedExampleDecoder =
    D.map3
        HighlightedExample
        (D.field "orig_url" Gradio.base64ImageDecoder)
        (D.field "highlighted_url" Gradio.base64ImageDecoder)
        (D.field "label" D.string)



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Image Classification"
    , body =
        [ Html.div
            [ Html.Attributes.class "flex flex-row" ]
            [ -- Image picker, left column
              Html.div
                [ Html.Attributes.class "flex flex-col flex-1" ]
                [ viewInputExample model.inputExample
                , Html.p [] [ Html.text "TODO: other example images here" ]
                ]

            -- SAE stuff, right column
            , Html.div
                [ Html.Attributes.class "flex flex-col flex-1" ]
                [ viewSaeActivations model.saeActivations ]
            ]
        ]
    }


viewInputExample : Requests.Requested Example -> Html.Html Msg
viewInputExample requestedExample =
    case requestedExample of
        Requests.Initial ->
            Html.p
                []
                [ Html.text "Loading initial example..." ]

        Requests.Loading ->
            Html.p
                []
                [ Html.text "Loading example..." ]

        Requests.Loaded example ->
            Html.img
                [ Html.Attributes.src (Gradio.base64ImageToString example.url) ]
                []

        Requests.Failed err ->
            viewErr err


viewSaeActivations : Requests.Requested (Dict.Dict String (List SaeActivation)) -> Html.Html Msg
viewSaeActivations requestedActivations =
    case requestedActivations of
        Requests.Initial ->
            Html.p
                [ Html.Attributes.class "italic" ]
                [ Html.text "Load an image to see SAE activations." ]

        Requests.Loading ->
            Html.p
                []
                [ Html.text "Loading SAE activations..." ]

        Requests.Loaded activations ->
            Html.div
                []
                (Dict.toList activations
                    |> List.map (uncurry viewModelSaeActivations)
                )

        Requests.Failed err ->
            viewErr err


viewModelSaeActivations : String -> List SaeActivation -> Html.Html Msg
viewModelSaeActivations model saeActivations =
    Html.div
        []
        [ Html.p
            []
            [ Html.text model ]
        , Html.div
            [ Html.Attributes.class "grid grid-cols-1 lg:grid-cols-2" ]
            (List.map
                viewSaeActivation
                saeActivations
            )
        ]


viewSaeActivation : SaeActivation -> Html.Html Msg
viewSaeActivation { latent, activations, examples } =
    Html.div []
        [ Html.details
            [ Html.Attributes.class "cursor-pointer rounded-lg border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200 dark:border-gray-700 dark:bg-gray-800 " ]
            [ Html.summary
                [ Html.Attributes.class "cursor-pointer select-none px-4 py-3 hover:bg-gray-50 text-gray-900" ]
                [ Html.span
                    []
                    [ Html.text ("Latent: " ++ String.fromInt latent) ]
                ]
            , Html.div
                [ Html.Attributes.class "p-1 text-gray-600 border-t border-gray-200 grid gap-1 grid-cols-2 md:grid-cols-4" ]
                (List.map viewHighlightedExample examples)
            ]
        ]


viewHighlightedExample : HighlightedExample -> Html.Html Msg
viewHighlightedExample { original, highlighted, label } =
    Html.div
        [ Html.Attributes.class "relative group" ]
        [ Html.img
            [ Html.Attributes.class "transition-opacity duration-100 group-hover:opacity-0"
            , Html.Attributes.src (Gradio.base64ImageToString original)
            ]
            []
        , Html.img
            [ Html.Attributes.class "absolute inset-0 opacity-0 transition-opacity duration-100 group-hover:opacity-100"
            , Html.Attributes.src (Gradio.base64ImageToString highlighted)
            ]
            []
        ]


viewErr : String -> Html.Html Msg
viewErr err =
    Html.div
        [ Html.Attributes.class "relative rounded-lg border border-red-200 bg-red-50 p-4 m-4" ]
        [ Html.button
            []
            []
        , Html.h3
            [ Html.Attributes.class "font-bold text-red-800" ]
            [ Html.text "Error" ]
        , Html.p
            [ Html.Attributes.class "text-red-700" ]
            [ Html.text err ]
        ]



-- HELPERS


uncurry : (a -> b -> c) -> ( a, b ) -> c
uncurry f ( a, b ) =
    f a b
